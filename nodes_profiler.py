import traceback
from datetime import datetime

import folder_paths
from comfy_api.latest import io

from .common import (append_history, find_history_entry, history_entry_options, log,
                     make_progress_cb, save_lora_profile)
from .llm_local import LocalLLM, tensor_to_base64_jpeg
from .nodes_local import advanced_model_inputs, build_model_config, model_input, sampling_inputs

SYSTEM_TEMPLATES = {
    "character": """You are profiling a character LoRA for a generative image model. The attached image was generated with the LoRA active under a neutral probe prompt. Extract the identity the LoRA reliably produces.

Rules:
- Describe only traits actually visible in the image; never invent appearance details.
- Cover gender presentation, apparent age, face shape and features, hair color and style, eye color, signature clothing or accessories, and recurring motifs.
- One compact factual paragraph in English. Ignore pose, composition, and lighting — those come from the probe prompt, not the LoRA.
- Output ONLY the profile paragraph. No explanations, no markdown fences.""",
    "style": """You are profiling a style LoRA for a generative image model. The attached image was generated with the LoRA active under a neutral probe prompt. Extract the visual style the LoRA reliably produces.

Rules:
- Describe only stylistic traits actually visible in the image; never invent.
- Cover medium and rendering technique, line work, palette tendencies, shading and texture, treatment of backgrounds, and overall mood.
- Ignore the specific subject matter — it comes from the probe prompt, not the LoRA.
- One compact factual paragraph in English. Output ONLY the profile paragraph. No explanations, no markdown fences.""",
}


class LoraProfilerLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        loras = folder_paths.get_filename_list("loras") or ["(no loras found)"]
        return io.Schema(
            node_id="LoraProfilerLocal",
            display_name="LoRA Profiler (Local GGUF)",
            category="prompt",
            description="Profile what a LoRA reliably produces: feed a probe image generated with the LoRA active and a neutral prompt, and the local vision model extracts its stable traits. The profile is stored under the LoRA's name in lora_contexts.json, where the Universal Prompt Enhancer's lora_profile input picks it up. Best with a single probe image. Unloads the model afterwards unless keep_loaded is on.",
            inputs=[
                io.Image.Input("image", tooltip="Probe image generated with the LoRA active and a neutral prompt."),
                io.Combo.Input("lora_name", options=loras, default=loras[0], tooltip="LoRA being profiled. The profile is stored under this exact name."),
                io.Combo.Input("probe_type", options=["character", "style"], default="character", tooltip="character: identity traits (face, hair, signature clothing). style: medium, line work, palette, shading, mood."),
                model_input(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=-1.0, min=-1.0, max=2.0, step=0.05, tooltip="-1 = auto: follow the Qwen3.8 official preset for the thinking mode (1.0 thinking / 0.7 non-thinking). >=0 = manual override; 0.3 restores the old factual default."),
                io.Int.Input("max_tokens", default=2048, min=64, max=32768, step=64),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Off: unload the model after generation so later nodes get the VRAM back. On: keep it resident for repeated runs."),
                *advanced_model_inputs(),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored run instead of calling the model. List refreshes when the node is created or the page reloads."),
                io.String.Input("trigger", default="", optional=True, tooltip="Activation term that wakes the LoRA up (usually the character/style name). Stored in the profile; the enhancer then keeps it verbatim in the final prompt. IMPORTANT: the probe image must have been generated with this term in the prompt, otherwise the LoRA stays dormant and the profile describes the base model instead."),
                *sampling_inputs(),
            ],
            outputs=[io.String.Output(display_name="lora_profile")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image, lora_name, probe_type, model, thinking, temperature, max_tokens, seed,
                keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe, history_entry="none", trigger="",
                top_p=-1.0, top_k=-1, presence_penalty=-1.0, reasoning_effort="low"):
        if history_entry != "none":
            e = find_history_entry(history_entry)
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(e.get("output", ""))

        if lora_name.startswith("("):
            raise ValueError("No LoRA found under models/loras.")

        config = build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort)
        sampling = LocalLLM.resolve_sampling(thinking == "enabled", temperature, top_p, top_k, presence_penalty)

        try:
            LocalLLM.load(config)
            if not LocalLLM.has_vision():
                raise ValueError(f"No *mmproj*.gguf found next to {model}; profiling needs a vision projector.")

            on_text = make_progress_cb(cls.hidden.unique_id)
            system = SYSTEM_TEMPLATES[probe_type]
            log(f"lora profile: lora={lora_name} type={probe_type} model={model} frames={image.shape[0]} thinking={thinking}")

            results = []
            for i, frame in enumerate(image):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Extract the {probe_type} profile from this probe image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tensor_to_base64_jpeg(frame)}"}},
                    ]},
                ]
                content = LocalLLM.generate(messages, max_tokens=max_tokens, sampling=sampling, seed=seed, on_text=on_text)
                log(f"frame {i + 1}/{image.shape[0]} ({len(content)} chars):\n{content[:1000]}")
                results.append(content.strip())
                LocalLLM.clear_context()  # prevent KV carry-over between frames

            output = "\n\n".join(results)
            save_lora_profile(lora_name, probe_type, output, trigger)
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "optimize",
                "task_type": f"profile/{lora_name}",
                "model": model,
                "thinking": thinking,
                "input": f"{image.shape[0]} probe frame(s), {probe_type}",
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
