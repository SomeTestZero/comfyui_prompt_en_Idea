import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, find_history_entry, history_entry_options, log, make_progress_cb
from .nodes_cloud import (
    ENHANCE_MAX_TOKENS,
    chat,
    image_part,
    model_combo,
    parse_model_option,
    require_vision,
)
from .skills import load_skill, scan_skills

# The official PE guides end in a JSON envelope (rewritten_prompt / wh_ratio /
# ratio_follow); the workflow's resolution node owns sizing, so the envelope is
# unwrapped to plain prompt text here.
QWEN21_SYSTEM = """You are a professional prompt engineer for the Qwen-Image-2.1 text-to-image and image-editing model. Rewrite the user's request into a final Qwen-Image-2.1 prompt that strictly follows the installed skill and the official prompt-rewriting guide below.

Rules:
- Task mode: {mode}. Follow the guide's register and structure for this mode exactly.
- The guide's JSON envelope (rewritten_prompt / wh_ratio / ratio_follow) does not apply here: output the rewritten prompt text itself, one continuous paragraph, and ignore the JSON fields. Everything the guide says about the content and formatting of rewritten_prompt still applies.
- Resolution and aspect ratio are set by separate workflow nodes: never mention resolution, aspect ratio, or pixel counts in the prompt.
- Output ONLY the final prompt. No explanations, no commentary, no markdown fences.{image_note}{background_note}

=== SKILL ===
{skill_body}

=== OFFICIAL PROMPT REWRITING GUIDE: {ref_names} ===
{ref_text}"""

# background=Transparent: the official RGBA wrapper is applied to the whole
# description and no backdrop is described (native transparent output).
RGBA_BACKGROUND_NOTE = (
    "\n- Background: transparent. Describe no backdrop of any kind, and wrap the ENTIRE description in the "
    "official RGBA wording verbatim: `This is an RGBA image with transparency. <description>. The image has "
    "alpha channel and the background is transparent.`"
)


def derive_mode(task_type, frames):
    """Auto: any image connected -> Edit (the edit prompt register), else T2I.
    Explicit modes pass through untouched — CharacterSheet stays CharacterSheet
    with or without images (the sheet guide handles both cases)."""
    if task_type == "Auto":
        return "Edit" if frames else "T2I"
    return task_type


def collect_frames(images):
    """Flattened slot order: the autogrow keys image_1..image_N map to the
    TextEncodeQwenImage21 slot and <imageN> tag of the same number."""
    frames = []
    if images:
        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            frames.extend(images[name])
    return frames


def build_user_text(prompt, frames):
    """(user_text, image_note) pair; with no text request the edit intent is
    reverse-inferred from the attached pictures."""
    user_text = prompt.strip()
    if not frames:
        return user_text, ""
    listing = ", ".join(f"<image{i + 1}>" for i in range(len(frames)))
    image_note = (
        f"\n- The user attached {len(frames)} image(s), given in the user message in order: {listing}. "
        "The tags map one-to-one onto the TextEncodeQwenImage21 image slots (image_1 = <image1>, ...), and the "
        "final prompt must address them with these tags (a single image is referred to naturally as \"the image\" "
        "instead). Describe only what is actually visible in them; never invent appearance details."
    )
    if user_text:
        user_text += f"\n\n{len(frames)} image(s) attached in order: {listing}."
    else:
        user_text = (
            f"No text request was provided. Reverse-infer the intent from the {len(frames)} attached image(s), "
            f"given in order: {listing}, and write the final prompt from the images alone, following the "
            "skill and guide."
        )
    return user_text, image_note


class QwenImage21PromptEnhancerCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = scan_skills()
        default_skill = "qwen-image21-prompt-writing" if "qwen-image21-prompt-writing" in skills else skills[0]
        return io.Schema(
            node_id="QwenImage21PromptEnhancerCloud",
            display_name="Qwen Image 2.1 Prompt Enhancer (Cloud API)",
            category="prompt",
            description="Rewrite a rough request into a Qwen-Image-2.1 prompt (text-to-image, instruction editing, or character reference sheet) with a multimodal cloud model, guided by the installed skill (qwen-image21-prompt-writing archives the official PE-T2I / PE-I2I prompt-rewriting system prompts plus a character-sheet composition guide). image_N sockets map to TextEncodeQwenImage21's same-numbered slots and the prompt addresses them as <image1>, <image2>, ... Needs a vision model when images are connected. Supports history_entry replay across both backends.",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("task_type", options=["Auto", "T2I", "Edit", "CharacterSheet"], default="Auto", tooltip="Auto: any image socket connected -> Edit, none -> T2I. The edit prompt register (instruction anchored on the input image(s)) fits any run with conditioning images, including reference-driven scene composition. CharacterSheet: build a multi-view character reference sheet (人设图/turnaround: front/side/back elevations + facial close-up, text-free on a flat solid backdrop) from scratch or from attached reference image(s) — identity anchored to the images when present."),
                io.Combo.Input("background", options=["Auto", "Transparent"], default="Auto", tooltip="Auto: normal backdrop (CharacterSheet mode auto-picks a solid colour that contrasts the character's colouring). Transparent: RGBA output — the prompt wraps in the official transparency wording and describes no backdrop."),
                io.Combo.Input("skill", options=skills, default=default_skill, tooltip="qwen-image21-prompt-writing = official Qwen-Image-2.1 prompt-rewriting guides (PE-T2I for text-to-image, PE-I2I for editing), routed by task mode."),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True, tooltip="Deep-thinking switch (thinking.type). Doubao Seed and DeepSeek honor it; GLM models always think and ignore this - reasoning_effort controls their depth."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, tooltip="Sampling temperature; other sampling knobs follow the provider's server defaults."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Sent to the API; cloud serving reproduces outputs on a best-effort basis only."),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True, tooltip="Thinking depth. Doubao: sent as reasoning_effort (auto = server default). GLM: low/medium/high map to low/high/max (always thinking). DeepSeek ignores it."),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("image", tooltip="Source/reference image; image_N maps to TextEncodeQwenImage21's images.image_N slot and to <imageN> in the rewritten prompt."),
                        prefix="image_", min=0, max=12,
                    ),
                    tooltip="Source and reference images in slot order; wire the same image to the same-numbered images.image_N on TextEncodeQwenImage21. Any connected image -> Edit mode. With an empty prompt, the edit intent is reverse-inferred from these images.",
                ),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored run (local or cloud) instead of calling the API. List refreshes when the node is created or the page reloads."),
                io.String.Input("api_key", default="", optional=True, tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
            ],
            outputs=[io.String.Output(display_name="enhanced_prompt")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, task_type, skill, model, thinking, temperature, seed,
                reasoning_effort="low", images=None, history_entry="none", api_key="",
                background="Auto"):
        if history_entry != "none":
            e = find_history_entry(history_entry)
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(e.get("output", ""))

        provider, model_id = parse_model_option(model)
        frames = collect_frames(images)
        mode = derive_mode(task_type, frames)
        require_vision(provider, model_id, frames)
        if not prompt.strip() and not frames:
            raise ValueError("prompt is empty and no reference images connected.")

        skill_body, refs = load_skill(skill, mode)
        user_text, image_note = build_user_text(prompt, frames)
        system = QWEN21_SYSTEM.format(
            mode=mode,
            image_note=image_note,
            background_note=RGBA_BACKGROUND_NOTE if background == "Transparent" else "",
            skill_body=skill_body,
            ref_names=" + ".join(n for n, _ in refs) or "none",
            ref_text="\n\n".join(t for _, t in refs),
        )
        user_content = [{"type": "text", "text": user_text}] + [image_part(f) for f in frames]

        try:
            log(f"qwen21 enhance: provider={provider} model={model_id} mode={mode} skill={skill} images={len(frames)} thinking={thinking} effort={reasoning_effort}")
            log(f"user prompt: {user_text[:500]}")
            log(f"system prompt:\n{system}")
            content = chat(
                provider, model_id, system, user_content,
                temperature=temperature, seed=seed, thinking=thinking == "enabled",
                effort_choice=reasoning_effort, api_key=api_key, max_tokens=ENHANCE_MAX_TOKENS,
                on_text=make_progress_cb(cls.hidden.unique_id),
            )
            log(f"response ({len(content)} chars):\n{content[:1000]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "optimize",
                "task_type": f"cloud/qwen21/{mode}",
                "model": f"{provider}/{model_id}",
                "thinking": thinking,
                "input": prompt.strip() or f"{len(frames)} image(s), no text",
                "output": content,
            })
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
