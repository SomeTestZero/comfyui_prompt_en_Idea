import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, find_history_entry, history_entry_options, log, lora_profile_options, make_progress_cb, read_lora_profiles
from .llm_local import LocalLLM, tensor_to_base64_jpeg
from .nodes_local import advanced_model_inputs, build_model_config, model_input, sampling_inputs
from .skills import load_skill, scan_skills

SYSTEM_TEMPLATE = """You are a professional prompt engineer for generative image and video models. Rewrite the user's request into a final generation prompt that strictly follows the installed skill guide below.

Rules:
- Follow the skill's structure, output format, and target-model parameter conventions exactly.
- Output ONLY the final prompt. No explanations, no commentary, no markdown fences.{image_note}{lora_note}

=== SKILL ===
{skill_body}

=== REFERENCES: {ref_names} ===
{ref_text}"""


def build_lora_note(lora_profile):
    if lora_profile == "none":
        return ""
    profile = read_lora_profiles().get(lora_profile)
    if profile is None:
        raise ValueError(f"LoRA profile not found (list may be stale, reselect it): {lora_profile}")
    context = profile.get("context", "")
    if profile.get("probe_type") == "style":
        note = (
            f"\n- The final image will be generated with the style LoRA \"{lora_profile}\" active. Its verified style traits:\n"
            f"{context}\n"
            "Apply this style to whatever the user asks: keep medium, palette, line work, and rendering consistent with the "
            "profile, and do not introduce a competing style or medium. Subject matter, composition, action, and lighting are "
            "yours to develop."
        )
    else:
        note = (
            f"\n- The final image will be generated with the character LoRA \"{lora_profile}\" active. Its verified identity traits:\n"
            f"{context}\n"
            "Lock the identity traits (face, eyes, hair, signature features) exactly as described so the character stays "
            "recognizable. The outfit and overall aesthetic in the profile are only her default look: keep them when the user "
            "does not specify otherwise, but when the user's request names a different style, era, outfit, or setting, the "
            "user's request wins — fuse the locked identity into the requested style instead of copying the default look."
        )
    trigger = profile.get("trigger", "")
    if trigger:
        note += (
            f'\n- The LoRA only activates when its trigger term appears in the prompt. The final prompt MUST begin with '
            f'"{trigger}" verbatim — do not translate, rewrite, or remove it.'
        )
    return note


class UniversalPromptEnhancerLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = scan_skills()
        default_skill = "krea2-prompt-writing" if "krea2-prompt-writing" in skills else skills[0]
        return io.Schema(
            node_id="UniversalPromptEnhancerLocal",
            display_name="Universal Prompt Enhancer (Local GGUF)",
            category="prompt",
            description="Rewrite a rough request into a structured prompt with a local multimodal GGUF model (llama.cpp), guided by whichever skill is installed under skills/ (e.g. krea2-prompt-writing). Model-agnostic companion to the H3 enhancer. With an empty prompt, the intent is reverse-inferred from the connected reference images. Select a lora_profile to keep the rewrite from fighting the LoRA you generate with. Unloads the model afterwards unless keep_loaded is on.",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("skill", options=skills, default=default_skill, tooltip="Skill folder under this node's skills/ directory. krea2-prompt-writing targets Krea 2 (RAW/Turbo); h3-prompt-writing targets MiniMax H3 video modes."),
                model_input(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=-1.0, min=-1.0, max=2.0, step=0.05, tooltip="-1 = auto: follow the Qwen3.8 official preset for the thinking mode (1.0 thinking / 0.7 non-thinking). >=0 = manual override."),
                io.Int.Input("max_tokens", default=-1, min=-1, max=32768, step=64, tooltip="-1 = no cap: generate to EOS. Thinking tokens come out of this budget before the answer, so a small cap cuts the answer off mid-reasoning (the run then fails instead of handing back raw thinking)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("keep_loaded", default=False, tooltip="Off: unload the model after generation so later nodes get the VRAM back. On: keep it resident for repeated runs."),
                *advanced_model_inputs(),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image", tooltip="Style/subject reference; order maps to <Picture 1>, <Picture 2>, ..."),
                        names=[f"image_{i}" for i in range(1, 9)],
                        min=0,
                    ),
                    tooltip="Optional reference images (style, palette, subject). Requires a *mmproj*.gguf next to the model.",
                ),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored run instead of calling the model. List refreshes when the node is created or the page reloads."),
                io.Combo.Input("lora_profile", options=lora_profile_options(), default="none", optional=True, tooltip="Verified trait profile of the LoRA the final generation will use (built by the LoRA Profiler node). The enhancer writes around these traits instead of inventing conflicting style/appearance details. List refreshes on page reload."),
                *sampling_inputs(),
            ],
            outputs=[io.String.Output(display_name="enhanced_prompt")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, skill, model, thinking, temperature, max_tokens, seed,
                keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe, reference_images=None,
                history_entry="none", lora_profile="none",
                top_p=-1.0, top_k=-1, presence_penalty=-1.0, reasoning_effort="low"):
        if history_entry != "none":
            e = find_history_entry(history_entry)
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(e.get("output", ""))

        frames = []
        if reference_images:
            for name in sorted(reference_images, key=lambda n: int(n.rsplit("_", 1)[-1])):
                for frame in reference_images[name]:
                    frames.append(frame)

        if not prompt.strip() and not frames:
            raise ValueError("prompt is empty and no reference images connected.")

        config = build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort)
        sampling = LocalLLM.resolve_sampling(thinking == "enabled", temperature, top_p, top_k, presence_penalty)

        try:
            LocalLLM.load(config)
            if frames and not LocalLLM.has_vision():
                raise ValueError(f"{len(frames)} image(s) connected, but no *mmproj*.gguf was found next to {model}; image input needs a vision projector.")

            skill_body, refs = load_skill(skill, mode="generic")
            on_text = make_progress_cb(cls.hidden.unique_id)
            log(f"universal enhance: skill={skill} model={model} images={len(frames)} thinking={thinking} lora_profile={lora_profile}")

            image_note = ""
            user_text = prompt.strip()
            if frames:
                listing = ", ".join(f"<Picture {i + 1}>" for i in range(len(frames)))
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

            system = SYSTEM_TEMPLATE.format(
                image_note=image_note,
                lora_note=build_lora_note(lora_profile),
                skill_body=skill_body,
                ref_names=" + ".join(n for n, _ in refs) or "none",
                ref_text="\n\n".join(t for _, t in refs),
            )
            user_content = [{"type": "text", "text": user_text}]
            for frame in frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{tensor_to_base64_jpeg(frame)}"},
                })
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]

            log(f"user prompt: {user_text[:500]}")
            log(f"system prompt:\n{system}")
            content = LocalLLM.generate(messages, max_tokens=max_tokens, sampling=sampling, seed=seed, on_text=on_text)
            log(f"response ({len(content)} chars):\n{content[:1000]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "optimize",
                "task_type": f"universal/{skill}",
                "model": model,
                "thinking": thinking,
                "lora_profile": lora_profile,
                "input": prompt.strip() or f"{len(frames)} image(s), no text",
                "output": content,
            })
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if keep_loaded:
                LocalLLM.clear_context()
            else:
                LocalLLM.unload()
