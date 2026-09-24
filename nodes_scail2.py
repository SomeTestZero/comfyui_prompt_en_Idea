"""SCAIL-2 helpers: dynamic segment planning for the auto-loop replacement
workflows, and LLM prompt generation on the same cloud provider surface as the
H3 cloud nodes."""

import json
import math
import re
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, log, make_progress_cb
from .nodes_cloud import (
    chat,
    image_part,
    model_combo,
    parse_model_option,
    require_vision,
)

PROMPT_MAX_TOKENS = 2048

SCAIL2_PROMPT_SYSTEM = """You write prompts for SCAIL-2 in-video character replacement (ComfyUI WanSCAILToVideo).
The driving video already supplies all motion, acting and camera work - the prompt must NOT mention actions, motion, poses, gestures or camera moves.
Output STRICT JSON only, no prose, keys exactly:
{"prompt": "...", "sam3_video_object": "...", "sam3_image_object": "..."}
- "prompt": 80-180 words in English describing the output scene: the replacement character's appearance from the reference image first (hair, face, build, clothing, accessories), then the environment and lighting from the driving frame. Present tense, static scene description only.
- "sam3_video_object": 2-5 comma-separated English tracking words for the person to replace in the driving frame (e.g. "woman, white tank top, denim shorts"). Every word must be visually verifiable in the frame.
- "sam3_image_object": 1-3 comma-separated English tracking words for the character in the reference image (usually "human"; for animals use the species).
Describe only what is visible in the attached images; never invent details, brands or identities."""


def _parse_result(content, manual_video, manual_image):
    """LLM reply -> (prompt, sam3_video_object, sam3_image_object). JSON contract
    with a text fallback: a broken reply still yields a usable prompt."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    body = fenced.group(1) if fenced else None
    if body is None:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            body = text[start:end + 1]
    if body:
        try:
            data = json.loads(body)
            return (
                str(data.get("prompt", "")).strip() or text,
                str(data.get("sam3_video_object", "")).strip() or manual_video,
                str(data.get("sam3_image_object", "")).strip() or manual_image,
            )
        except json.JSONDecodeError:
            pass
    return text, manual_video, manual_image


class SCAIL2SegmentPlan(io.ComfyNode):
    """Frame count -> per-iteration chunk offsets for Start Loop (List mode).
    Emits one offset per continuation segment: segments 2..N at step
    (chunk - overlap) frames, matching the official Extend bookkeeping."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2SegmentPlan",
            display_name="SCAIL-2 Segment Plan 自动分段",
            category="utilities",
            search_aliases=["scail", "loop", "segment"],
            inputs=[
                io.Int.Input("frame_count", default=200, min=1, tooltip="Total driving video frames - wire GetImageSize batch_size."),
                io.Int.Input("chunk", default=81, min=2, tooltip="Frames generated per segment (SCAIL-2 trained at 81)."),
                io.Int.Input("overlap", default=5, min=1, tooltip="Anchored tail frames per segment (trained at 5); the loop step is chunk - overlap."),
            ],
            outputs=[
                io.AnyType.Output(is_output_list=True, display_name="offsets",
                                  tooltip="Chunk start offsets for continuation segments - wire into Start Loop's list (List mode). Single-segment videos produce an empty list."),
            ],
        )

    @classmethod
    def execute(cls, frame_count, chunk, overlap) -> io.NodeOutput:
        step = chunk - overlap
        if step <= 0:
            raise ValueError(f"chunk ({chunk}) must exceed overlap ({overlap})")
        n_segments = math.ceil(frame_count / step)
        offsets = [step * i for i in range(1, n_segments)]
        log(f"segment plan: frames={frame_count} chunk={chunk} step={step} -> {n_segments} segment(s), offsets={offsets}")
        return io.NodeOutput(offsets)


class SCAIL2PromptGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2PromptGenerator",
            display_name="SCAIL-2 Prompt Generator (Cloud API) 提示词生成",
            category="prompt",
            description="Writes the SCAIL-2 character-replacement prompt bundle from a short idea plus the reference/driving images: the English scene prompt (appearance + background, no actions - the driving video supplies motion) and the SAM3 tracking words for both videos. Same cloud provider surface as the H3 cloud nodes; disabled = passthrough of the manual inputs.",
            inputs=[
                io.String.Input("hint", multiline=True, default="", tooltip="Optional extra direction in any language (mood, lighting, what to keep from the scene); everything it does not specify is taken from the images."),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=0.5, min=0.0, max=2.0, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True),
                io.Image.Input("reference_image", optional=True, tooltip="Replacement character image (first frame is used). Match the sampler's reference image."),
                io.Image.Input("driving_frame", optional=True, tooltip="A frame of the driving video (first frame is used) - supplies the background description and the replace-target tracking words."),
                io.Boolean.Input("enabled", default=True, tooltip="False = skip the API call and pass manual_prompt / manual_sam3_* through unchanged."),
                io.String.Input("manual_prompt", default="", multiline=True, tooltip="Fallback prompt used when enabled is false."),
                io.String.Input("manual_sam3_video", default="person", tooltip="Fallback replace-target tracking words (e.g. \"girl, handkerchief\")."),
                io.String.Input("manual_sam3_image", default="human", tooltip="Fallback reference tracking words."),
                io.String.Input("api_key", default="", optional=True, tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="sam3_video_object"),
                io.String.Output(display_name="sam3_image_object"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, hint, model, thinking, temperature, seed, reasoning_effort,
                enabled, manual_prompt, manual_sam3_video, manual_sam3_image,
                reference_image=None, driving_frame=None, api_key=""):
        if not enabled:
            return io.NodeOutput(manual_prompt, manual_sam3_video, manual_sam3_image)
        provider, model_id = parse_model_option(model)
        frames = []
        if reference_image is not None:
            frames.append((reference_image[0], "reference image"))
        if driving_frame is not None:
            frames.append((driving_frame[0], "driving video frame"))
        require_vision(provider, model_id, frames)

        labels = "\n".join(f"image {i + 1} = {label}" for i, (_, label) in enumerate(frames))
        user_text = (
            (hint.strip() or "Describe the replacement character and scene.") +
            (f"\n\nAttached images:\n{labels}" if labels else "\n\nNo images attached - fall back to the hint only and keep tracking words generic.")
        )
        user_content = [{"type": "text", "text": user_text}] + [image_part(f) for f, _ in frames]
        try:
            log(f"scail2 prompt: provider={provider} model={model_id} images={len(frames)} seed={seed} thinking={thinking} hint={hint.strip()[:100]!r}")
            content = chat(
                provider, model_id, SCAIL2_PROMPT_SYSTEM, user_content,
                temperature=temperature, seed=seed, thinking=thinking == "enabled",
                effort_choice=reasoning_effort, api_key=api_key, max_tokens=PROMPT_MAX_TOKENS,
                on_text=make_progress_cb(cls.hidden.unique_id),
            )
            prompt, sam3_video, sam3_image = _parse_result(content, manual_sam3_video, manual_sam3_image)
            log(f"scail2 prompt ({len(prompt)} chars): {prompt[:300]!r}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "scail2-prompt",
                "task_type": f"cloud/scail2/{len(frames)}img",
                "model": f"{provider}/{model_id}",
                "seed": seed,
                "input": hint.strip() or f"{len(frames)} image(s)",
                "output": json.dumps({"prompt": prompt, "sam3_video_object": sam3_video, "sam3_image_object": sam3_image}, ensure_ascii=False),
            })
            return io.NodeOutput(prompt, sam3_video, sam3_image)
        except Exception as e:
            if manual_prompt.strip():
                log(f"scail2 prompt failed ({e}); falling back to manual inputs")
                return io.NodeOutput(manual_prompt, manual_sam3_video, manual_sam3_image)
            raise
