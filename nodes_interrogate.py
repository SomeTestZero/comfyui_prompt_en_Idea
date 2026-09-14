import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, find_history_entry, history_entry_options, log, make_progress_cb
from .llm_local import INTERROGATE_LONG_EDGE, LocalLLM, tensor_to_base64_jpeg
from .nodes_local import advanced_model_inputs, build_model_config, model_input, sampling_inputs
from .skills import load_skill, scan_skills

INTERROGATE_SYSTEM = """You are a forensic visual analyst for generative image models. Analyze the attached image as visual evidence and produce a reverse prompt strictly following the installed skill guide below.

Rules:
- Describe only what is actually visible in the image; never invent appearance details, identities, brands, or camera metadata.
- Follow the skill's output format exactly. Output ONLY the final description or tag list. No explanations, no commentary, no markdown fences.

=== SKILL ===
{skill_body}

=== REFERENCES: {ref_names} ===
{ref_text}"""

INTERROGATE_USER_INSTRUCTIONS = """

=== USER INSTRUCTIONS ===
The user steered this run with the request below. Follow it when deciding what to describe or leave out, but never let it make you invent content or break the skill's output format.

{custom_instruction}"""


class UniversalImageInterrogatorLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = scan_skills()
        default_skill = "img2prompt-natural" if "img2prompt-natural" in skills else skills[0]
        return io.Schema(
            node_id="UniversalImageInterrogatorLocal",
            display_name="Universal Image Interrogator (Local GGUF)",
            category="prompt",
            description="Reverse-prompt an image with a local multimodal GGUF model (llama.cpp). Skill selects the output style: img2prompt-natural (natural-language paragraph for Krea 2/FLUX-era models) or img2prompt-tags (booru tag list for SD1.5/Pony/SDXL-era models). custom_instruction steers every frame (e.g. skip watermarks). Batches are described frame by frame. Unloads the model afterwards unless keep_loaded is on.",
            inputs=[
                io.Image.Input("image", tooltip="Image(s) to reverse-prompt. A batch is processed frame by frame and the results are joined."),
                io.Combo.Input("skill", options=skills, default=default_skill, tooltip="img2prompt-natural: natural-language prompt. img2prompt-tags: booru-style comma-separated tag list."),
                io.String.Input("custom_instruction", multiline=True, default="", optional=True, tooltip="Steering request applied to every frame, e.g. 'Do not mention watermarks, logos, or subtitles.' Empty = plain skill behavior."),
                model_input(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=-1.0, min=-1.0, max=2.0, step=0.05, tooltip="-1 = auto: follow the Qwen3.8 official preset for the thinking mode (1.0 thinking / 0.7 non-thinking). >=0 = manual override; 0.3 restores the old factual default."),
                io.Int.Input("max_tokens", default=-1, min=-1, max=32768, step=64, tooltip="-1 = no cap: generate to EOS. Thinking tokens come out of this budget before the answer, so a small cap cuts the answer off mid-reasoning (the run then fails instead of handing back raw thinking)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("keep_loaded", default=False, tooltip="Off: unload the model after generation so later nodes get the VRAM back. On: keep it resident for repeated runs."),
                *advanced_model_inputs(),
                io.Combo.Input("history_entry", options=history_entry_options(kind="interrogate"), default="none", optional=True, tooltip="Replay a stored interrogation (local or cloud) instead of calling the model. List refreshes when the node is created or the page reloads."),
                *sampling_inputs(),
            ],
            outputs=[io.String.Output(display_name="description")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image, skill, model, thinking, temperature, max_tokens, seed,
                keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe, custom_instruction="",
                history_entry="none", top_p=-1.0, top_k=-1, presence_penalty=-1.0,
                reasoning_effort="low"):
        custom_instruction = custom_instruction.strip()
        if history_entry != "none":
            e = find_history_entry(history_entry, kind="interrogate")
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(e.get("output", ""))

        config = build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort)
        sampling = LocalLLM.resolve_sampling(thinking == "enabled", temperature, top_p, top_k, presence_penalty)

        try:
            LocalLLM.load(config)
            if not LocalLLM.has_vision():
                raise ValueError(f"No *mmproj*.gguf found next to {model}; image interrogation needs a vision projector.")

            skill_body, refs = load_skill(skill, mode="generic")
            on_text = make_progress_cb(cls.hidden.unique_id)
            log(f"interrogate: skill={skill} model={model} frames={image.shape[0]} thinking={thinking} instruction={'yes' if custom_instruction else 'no'}")

            system = INTERROGATE_SYSTEM.format(
                skill_body=skill_body,
                ref_names=" + ".join(n for n, _ in refs) or "none",
                ref_text="\n\n".join(t for _, t in refs),
            )
            if custom_instruction:
                system += INTERROGATE_USER_INSTRUCTIONS.format(custom_instruction=custom_instruction)

            results = []
            for i, frame in enumerate(image):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Reverse-prompt this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tensor_to_base64_jpeg(frame, max_long_edge=INTERROGATE_LONG_EDGE)}"}},
                    ]},
                ]
                content = LocalLLM.generate(messages, max_tokens=max_tokens, sampling=sampling, seed=seed, on_text=on_text)
                log(f"frame {i + 1}/{image.shape[0]} ({len(content)} chars):\n{content[:1000]}")
                results.append(content.strip())
                LocalLLM.clear_context()  # prevent KV carry-over between frames

            output = "\n\n---\n\n".join(results)
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "interrogate",
                "task_type": f"local/interrogate/{skill}",
                "instruction": custom_instruction,
                "model": model,
                "thinking": thinking,
                "input": f"{image.shape[0]} frame(s)",
                "output": output,
            })
            return io.NodeOutput(output)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if keep_loaded:
                LocalLLM.clear_context()
            else:
                LocalLLM.unload()
