import os
import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, find_history_entry, history_entry_options, log, make_progress_cb
from .llm_local import LocalLLM, find_mmproj, list_gguf_models, register_llm_folder, tensor_to_base64_jpeg
from .nodes_api import TRANSLATE_SYSTEM
from .skills import MODES, load_skill, scan_skills

import folder_paths

SYSTEM_TEMPLATE = """You are a professional prompt engineer for the MiniMax H3 video generation model. Rewrite the user's request into a final H3 generation prompt that strictly follows the official skill and writing guide below.

Rules:
- Task mode: {mode}. Follow the guide's structure, section order, reference labels, and timing notation for this mode exactly.
- Write rewrite sections in English; preserve dialogue, lyrics, and visible scene text in their original language.
- Output ONLY the final prompt. No explanations, no commentary, no markdown fences.{image_note}

=== SKILL ===
{skill_body}

=== WRITING GUIDE: {ref_names} ===
{ref_text}"""

def derive_mode(first_frame, last_frame, reference_images):
    """Deterministic mode from which image sockets are connected."""
    if reference_images:
        return "Ref2VA"
    if first_frame is not None and last_frame is not None:
        return "FL2VA"
    if first_frame is not None:
        return "I2VA"
    if last_frame is not None:
        return "L2VA"
    return "T2VA"


def model_input():
    """Shared model combo; identical default across nodes so a chained node
    reuses the already-loaded model instead of reloading."""
    models = list_gguf_models() or ["(no .gguf found under models/LLM)"]
    default_model = (
        next((m for m in models if "qwen3.8" in m.lower()), None)
        or next((m for m in models if "35b" in m.lower()), None)
        or models[0]
    )
    return io.Combo.Input("model", options=models, default=default_model, tooltip="GGUF from models/LLM; a *mmproj*.gguf next to it enables image inputs.")


def advanced_model_inputs():
    return [
        io.Int.Input("n_ctx", default=65536, min=0, max=262144, step=1024, advanced=True, tooltip="Context size. Default 65536: KV ~4GiB at f16 (~64KiB/token; only the 16 full-attention layers carry KV, so with n_gpu_layers=32 half of it sits in RAM). 0 = the model's full native context (262144 for Qwen3.8) — only viable when most weights sit in system RAM, e.g. MoE + n_cpu_moe. Budget: skill guide + images (~1k each) + output."),
        io.Int.Input("n_gpu_layers", default=32, min=-1, max=200, advanced=True, tooltip="Layers on GPU, counted from the last one (the hybrid Qwen3.8-27B has 66 incl. the output layer). 32 + n_ctx=65536 leaves ~3.6GiB free on a 16GB card at ~7 t/s; 48 fills VRAM completely and collapses under 1 t/s once mmproj/ComfyUI are resident (WDDM paging). Lower toward 24 when other models stay resident. -1 = all layers on GPU (only viable for MoE + n_cpu_moe)."),
        io.Int.Input("n_cpu_moe", default=99, min=0, max=200, advanced=True, tooltip="Keep MoE expert weights in system RAM (0 = all on GPU). 99 = all experts on CPU, required for 35B-A3B on 16GB VRAM; harmless for dense models."),
    ]


def sampling_inputs():
    """Appended sampling widgets (see the positional-values note below). A value
    of -1 means "auto": follow the Qwen3.8 official preset of the current thinking
    mode — 1.0/0.95/20/0.0 thinking, 0.7/0.8/20/1.5 non-thinking. A manual value
    overrides just that parameter."""
    return [
        io.Float.Input("top_p", default=-1.0, min=-1.0, max=1.0, step=0.05, advanced=True, tooltip="-1 = auto (Qwen3.8 official: 0.95 thinking / 0.8 non-thinking). >=0 = manual override."),
        io.Int.Input("top_k", default=-1, min=-1, max=500, advanced=True, tooltip="-1 = auto (Qwen3.8 official: 20 both modes). >=0 = manual override."),
        io.Float.Input("presence_penalty", default=-1.0, min=-1.0, max=2.0, step=0.05, advanced=True, tooltip="-1 = auto (Qwen3.8 official: 0.0 thinking / 1.5 non-thinking). >=0 = manual override."),
        io.Combo.Input("reasoning_effort", options=["xhigh", "medium", "low"], default="low", advanced=True, tooltip="Qwen3.8 thinking depth, only with thinking enabled (default low = short reasoning, fastest; xhigh = the model's official default). Passed into the chat template, but the mmproj vision path renders with llama-cpp-python's built-in Qwen template, which has no such variable - so on image runs it currently has no effect (verified: low and xhigh produce identical output)."),
    ]


# NOTE: widget values in saved workflows are positional — never reorder,
# insert, or remove existing inputs on a released node; append only.
def build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort="low"):
    if model.startswith("("):
        raise ValueError("No GGUF model found under models/LLM.")
    register_llm_folder()
    model = model.replace("/", os.sep)  # tolerate forward slashes from cross-platform workflows
    return {
        "model_path": folder_paths.get_full_path("LLM", model),
        "mmproj": find_mmproj(model),
        "thinking": thinking == "enabled",
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "n_cpu_moe": n_cpu_moe,
        "reasoning_effort": reasoning_effort,
    }


def collect_frames(mode, first_frame, last_frame, reference_images):
    """(tensor, role) pairs in <Picture N> order; T2VA returns an empty list."""
    frames = []
    if mode == "Ref2VA":
        if reference_images:
            for name in sorted(reference_images, key=lambda n: int(n.rsplit("_", 1)[-1])):
                for frame in reference_images[name]:
                    frames.append((frame, "reference"))
        if first_frame is not None:
            frames.append((first_frame[0], "first-frame anchor"))
        if last_frame is not None:
            frames.append((last_frame[0], "last-frame anchor"))
        if not frames:
            raise ValueError("Ref2VA needs at least one reference image or keyframe connected.")
    elif mode == "I2VA":
        if first_frame is None:
            raise ValueError("I2VA needs first_frame connected.")
        frames = [(first_frame[0], "first frame")]
    elif mode == "FL2VA":
        if first_frame is None or last_frame is None:
            raise ValueError("FL2VA needs both first_frame and last_frame connected.")
        frames = [(first_frame[0], "first frame"), (last_frame[0], "last frame")]
    elif mode == "L2VA":
        if last_frame is None:
            raise ValueError("L2VA needs last_frame connected.")
        frames = [(last_frame[0], "last frame")]
    return frames

def build_enhance_user_text(prompt, frames):
    """(user_text, image_note) pair for the enhancer system template; with no
    text request the intent is reverse-inferred from the attached pictures."""
    user_text = prompt.strip()
    if frames:
        listing = ", ".join(f"<Picture {i + 1}> ({role})" for i, (_, role) in enumerate(frames))
        image_note = (
            f"\n- The user attached {len(frames)} image(s), given in the user message in order: {listing}. "
            "Describe only what is actually visible in them; never invent appearance details."
        )
        if user_text:
            user_text += f"\n\n{len(frames)} image(s) attached in order: {listing}."
        else:
            user_text = (
                f"No text request was provided. Reverse-infer the intent from the {len(frames)} attached image(s), "
                f"given in order: {listing}. Describe what is actually visible in them and write the final "
                "generation prompt from the images alone, following the skill's structure."
            )
        return user_text, image_note
    return user_text, ""


class H3PromptEnhancerLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = scan_skills()
        return io.Schema(
            node_id="H3PromptEnhancerLocal",
            display_name="H3 Prompt Enhancer (Local GGUF)",
            category="prompt",
            description="Rewrite a rough request into a structured MiniMax H3 prompt with a local multimodal GGUF model (llama.cpp), guided by an installed skill. Supports multiple reference images; with an empty prompt, the intent is reverse-inferred from the connected images/keyframes. Unloads the model afterwards unless keep_loaded is on.",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("task_type", options=MODES, default="Auto", tooltip="Auto: derived from which image sockets are connected — reference images -> Ref2VA, first+last frame -> FL2VA, first only -> I2VA, last only -> L2VA, none -> T2VA."),
                io.Combo.Input("skill", options=skills, default="h3-prompt-writing" if "h3-prompt-writing" in skills else skills[0]),
                model_input(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=-1.0, min=-1.0, max=2.0, step=0.05, tooltip="-1 = auto: follow the Qwen3.8 official preset for the thinking mode (1.0 thinking / 0.7 non-thinking). >=0 = manual override."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("keep_loaded", default=False, tooltip="Off: unload the model after generation so later nodes get the VRAM back. On: keep it resident for repeated runs."),
                *advanced_model_inputs(),
                io.Image.Input("first_frame", optional=True, tooltip="First keyframe image. With last_frame -> FL2VA; alone -> I2VA."),
                io.Image.Input("last_frame", optional=True, tooltip="Last keyframe image. Alone -> L2VA."),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("ref_image", tooltip="Subject/style reference; ref_image_N maps to <Picture N+1>, same as the MiniMax H3 Reference to Video socket of the same name."),
                        prefix="ref_image_", min=0, max=9,
                    ),
                    tooltip="Reference images (characters, style, props); socket names mirror the MiniMax H3 Reference to Video node — wire the same image to the same-numbered ref_image_N on both. Any connected reference -> Ref2VA; keyframes connected alongside are appended as anchor Pictures.",
                ),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored run instead of calling the model. List refreshes when the node is created or the page reloads."),
                *sampling_inputs(),
            ],
            outputs=[
                io.String.Output(display_name="enhanced_prompt"),
                io.String.Output(display_name="detected_mode"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, task_type, skill, model, thinking, temperature, seed,
                keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe, first_frame=None, last_frame=None,
                reference_images=None, history_entry="none",
                top_p=-1.0, top_k=-1, presence_penalty=-1.0, reasoning_effort="low"):
        if history_entry != "none":
            e = find_history_entry(history_entry)
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            stored = str(e.get("task_type", ""))
            replay_mode = stored.removeprefix("local/") if stored.startswith("local/") else ("T2VA" if task_type == "Auto" else task_type)
            log(f"history replay: {history_entry}")
            return io.NodeOutput(e.get("output", ""), replay_mode)

        mode = derive_mode(first_frame, last_frame, reference_images) if task_type == "Auto" else task_type

        frames = collect_frames(mode, first_frame, last_frame, reference_images)

        if not prompt.strip() and not frames:
            raise ValueError("prompt is empty and no reference images connected.")

        config = build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort)
        sampling = LocalLLM.resolve_sampling(thinking == "enabled", temperature, top_p, top_k, presence_penalty)

        try:
            LocalLLM.load(config)
            if frames and not LocalLLM.has_vision():
                raise ValueError(f"{len(frames)} image(s) connected, but no *mmproj*.gguf was found next to {model}; image input needs a vision projector.")

            skill_body, refs = load_skill(skill, mode)
            on_text = make_progress_cb(cls.hidden.unique_id)
            log(f"mode: {mode} (task_type={task_type})")

            user_text, image_note = build_enhance_user_text(prompt, frames)

            system = SYSTEM_TEMPLATE.format(
                mode=mode,
                image_note=image_note,
                skill_body=skill_body,
                ref_names=" + ".join(n for n, _ in refs) or "none",
                ref_text="\n\n".join(t for _, t in refs),
            )
            user_content = [{"type": "text", "text": user_text}]
            for frame, _ in frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{tensor_to_base64_jpeg(frame)}"},
                })
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]

            log(f"local enhance: mode={mode} skill={skill} model={model} images={len(frames)} thinking={thinking}")
            log(f"user prompt: {user_text[:500]}")
            log(f"system prompt:\n{system}")
            content = LocalLLM.generate(messages, max_tokens=-1, sampling=sampling, seed=seed, on_text=on_text)
            log(f"response ({len(content)} chars):\n{content[:1000]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "optimize",
                "task_type": f"local/{mode}",
                "model": model,
                "thinking": thinking,
                "input": prompt.strip() or f"{len(frames)} image(s), no text",
                "output": content,
            })
            return io.NodeOutput(content, mode)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if keep_loaded:
                LocalLLM.clear_context()
            else:
                LocalLLM.unload()


class H3TranslatorLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3TranslatorLocal",
            display_name="H3 Translator (Local GGUF) 审核翻译",
            category="prompt",
            description="Translate a prompt for quick review with the local GGUF model. Same model-config defaults as the enhancer: wire it after an enhancer running with keep_loaded=True and the model is loaded only once, then unloaded here.",
            inputs=[
                io.String.Input("text", multiline=True, default="", tooltip="Linked input hides the box; widget kept (not force_input) so saved workflows round-trip correctly."),
                io.Combo.Input("target_lang", options=["中文", "English"], default="中文"),
                model_input(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=-1.0, min=-1.0, max=2.0, step=0.05, tooltip="-1 = auto: follow the Qwen3.8 official preset for the thinking mode (1.0 thinking / 0.7 non-thinking). >=0 = manual override; 0.3 restores the old default."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("keep_loaded", default=False, tooltip="Off: unload the model after translation so later nodes get the VRAM back."),
                *advanced_model_inputs(),
                *sampling_inputs(),
            ],
            outputs=[io.String.Output(display_name="translated_text")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, text, target_lang, model, thinking, temperature, seed,
                keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe,
                top_p=-1.0, top_k=-1, presence_penalty=-1.0, reasoning_effort="low"):
        if not text.strip():
            log("local translate: empty input, passing through")
            return io.NodeOutput("")

        config = build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort)
        sampling = LocalLLM.resolve_sampling(thinking == "enabled", temperature, top_p, top_k, presence_penalty)
        try:
            LocalLLM.load(config)
            system = TRANSLATE_SYSTEM.format(target=target_lang)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": text.strip()},
            ]
            log(f"local translate: target={target_lang} model={model}")
            log(f"input text: {text.strip()[:500]}")
            content = LocalLLM.generate(
                messages, max_tokens=-1, sampling=sampling, seed=seed,
                on_text=make_progress_cb(cls.hidden.unique_id),
            )
            log(f"translation ({len(content)} chars):\n{content[:1000]}")
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if keep_loaded:
                LocalLLM.clear_context()
            else:
                LocalLLM.unload()


class H3LLMUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LLMUnload",
            display_name="Unload Local LLM (H3)",
            category="prompt",
            description="Unload the local GGUF model to free VRAM. Pass-through, wire it after a keep_loaded run.",
            inputs=[io.AnyType.Input("any_input", optional=True)],
            outputs=[io.AnyType.Output(display_name="any_output")],
        )

    @classmethod
    def execute(cls, any_input=None):
        LocalLLM.unload()
        return io.NodeOutput(any_input)
