import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, find_history_entry, history_entry_options, log, make_progress_cb
from .llm_local import LocalLLM
from .moss_prompt import TASK_TYPES, build_system, build_user_text, postprocess, validate_canvas
from .nodes_cloud import ENHANCE_MAX_TOKENS, chat, model_combo, parse_model_option
from .nodes_local import advanced_model_inputs, build_model_config, model_input, sampling_inputs
from .skills import load_skill, scan_skills
from .categories import PROMPT_ENHANCE

MOSS_SKILL = "moss-image-prompt-writing"

def task_type_input():
    return io.Combo.Input(
        "task_type", display_name="任务类型", options=list(TASK_TYPES), default="文生图",
        tooltip="文生图：一张图的完整描述（MOSS 的主流用法）。版式布局：海报/多格/精确排字，用 bbox 逐元素定位；"
                "输出自动编译成模型要的 abs_centralized/yxyx JSON，宽高取自本节点的 width/height（工作流里与 latent 同源）。",
    )


def canvas_inputs():
    return [
        io.Int.Input("width", display_name="画布宽", default=1024, min=16, max=16384, step=16,
                     tooltip="版式布局模式的画布宽度（也是编译 bbox 的坐标系）；请与采样 latent 的宽同源。"),
        io.Int.Input("height", display_name="画布高", default=1024, min=16, max=16384, step=16,
                     tooltip="版式布局模式的画布高度（也是编译 bbox 的坐标系）；请与采样 latent 的高同源。"),
    ]


def _resolve(prompt, task_type, width, height, skill):
    mode = TASK_TYPES.get(task_type, task_type)
    validate_canvas(width, height)
    skill_body, refs = load_skill(skill, mode)
    system = build_system(skill_body, refs, mode, width, height)
    user_text = build_user_text(prompt, mode, width, height)
    return mode, system, user_text


class MOSSPromptEnhancerLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = scan_skills()
        default_skill = MOSS_SKILL if MOSS_SKILL in skills else skills[0]
        return io.Schema(
            node_id="MOSSPromptEnhancerLocal",
            display_name="[yyp] MOSS Prompt Enhancer (Local GGUF)",
            category=PROMPT_ENHANCE,
            description="Rewrite a rough idea into a MOSS-Image prompt with a local multimodal GGUF model, guided by the "
                        "official moss-image-prompt-enhancer skill (archived under skills/). Caption mode writes a "
                        "finished-image description in the style MOSS was trained on; Layout mode asks for a bbox draft "
                        "and compiles it to the abs_centralized/yxyx JSON the model expects, validating the boxes against "
                        "the canvas. Unloads the model afterwards unless keep_loaded is on.",
            inputs=[
                io.String.Input("prompt", multiline=True, default="", tooltip="粗略想法（中文即可）；写清要出现的文字及其样子。"),
                task_type_input(),
                *canvas_inputs(),
                io.Combo.Input("skill", options=skills, default=default_skill,
                               tooltip="moss-image-prompt-writing = 官方 MOSS-Image Prompt Enhancer skill（caption 规则 + layout 草案规范）。"),
                model_input(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=-1.0, min=-1.0, max=2.0, step=0.05,
                               tooltip="-1 = auto: follow the Qwen3.8 official preset for the thinking mode (1.0 thinking / 0.7 non-thinking). >=0 = manual override."),
                io.Int.Input("max_tokens", default=-1, min=-1, max=32768, step=64,
                             tooltip="-1 = no cap: generate to EOS. Thinking tokens come out of this budget before the answer, so a small cap cuts the answer off mid-reasoning (the run then fails instead of handing back raw thinking)."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("keep_loaded", default=False,
                                 tooltip="Off: unload the model after generation so the MOSS DiT and text encoder get the VRAM back. On: keep it resident for repeated runs."),
                *advanced_model_inputs(),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True,
                               tooltip="Replay a stored run instead of calling the model. List refreshes when the node is created or the page reloads."),
                *sampling_inputs(),
            ],
            outputs=[io.String.Output(display_name="enhanced_prompt")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, task_type, width, height, skill, model, thinking, temperature, max_tokens,
                seed, keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe, history_entry="none",
                top_p=-1.0, top_k=-1, presence_penalty=-1.0, reasoning_effort="low"):
        if history_entry != "none":
            entry = find_history_entry(history_entry)
            if entry is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(entry.get("output", ""))

        mode, system, user_text = _resolve(prompt, task_type, width, height, skill)
        config = build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort)
        sampling = LocalLLM.resolve_sampling(thinking == "enabled", temperature, top_p, top_k, presence_penalty)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

        try:
            LocalLLM.load(config)
            on_text = make_progress_cb(cls.hidden.unique_id)
            log(f"moss enhance: mode={mode} canvas={width}x{height} skill={skill} model={model} thinking={thinking}")
            log(f"user prompt: {user_text[:500]}")
            log(f"system prompt:\n{system}")
            content = LocalLLM.generate(messages, max_tokens=max_tokens, sampling=sampling, seed=seed, on_text=on_text)
            log(f"raw response ({len(content)} chars):\n{content[:1000]}")
            result = postprocess(content, mode, width, height)
            log(f"final prompt ({len(result)} chars):\n{result[:1000]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "optimize",
                "task_type": f"local/moss/{mode.lower()}",
                "model": model,
                "thinking": thinking,
                "canvas": f"{width}x{height}",
                "input": prompt.strip(),
                "output": result,
            })
            return io.NodeOutput(result)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if keep_loaded:
                LocalLLM.clear_context()
            else:
                LocalLLM.unload()


class MOSSPromptEnhancerCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = scan_skills()
        default_skill = MOSS_SKILL if MOSS_SKILL in skills else skills[0]
        return io.Schema(
            node_id="MOSSPromptEnhancerCloud",
            display_name="[yyp] MOSS Prompt Enhancer (Cloud API)",
            category=PROMPT_ENHANCE,
            description="Cloud variant of the MOSS Prompt Enhancer: the same official skill, caption rules and Layout draft "
                        "compiler, written by an OpenAI-compatible multimodal cloud model. Caption mode outputs a finished-image "
                        "description; Layout mode outputs the compiled abs_centralized/yxyx JSON prompt for the canvas set on "
                        "the node. Supports history_entry replay across both backends.",
            inputs=[
                io.String.Input("prompt", multiline=True, default="", tooltip="粗略想法（中文即可）；写清要出现的文字及其样子。"),
                task_type_input(),
                *canvas_inputs(),
                io.Combo.Input("skill", options=skills, default=default_skill,
                               tooltip="moss-image-prompt-writing = 官方 MOSS-Image Prompt Enhancer skill（caption 规则 + layout 草案规范）。"),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True,
                               tooltip="Deep-thinking switch (thinking.type). Doubao Seed and DeepSeek honor it; GLM models always think and ignore it - reasoning_effort controls their depth."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05,
                               tooltip="Sampling temperature; other sampling knobs follow the provider's server defaults."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Sent to the API; cloud serving reproduces outputs on a best-effort basis only."),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True,
                               tooltip="Thinking depth. Doubao: sent as reasoning_effort (auto = server default). GLM: low/medium/high map to low/high/max (always thinking). DeepSeek ignores it."),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True,
                               tooltip="Replay a stored run (local or cloud) instead of calling the API. List refreshes when the node is created or the page reloads."),
                io.String.Input("api_key", default="", optional=True,
                                tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
            ],
            outputs=[io.String.Output(display_name="enhanced_prompt")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, task_type, width, height, skill, model, thinking, temperature, seed,
                reasoning_effort="low", history_entry="none", api_key=""):
        if history_entry != "none":
            entry = find_history_entry(history_entry)
            if entry is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(entry.get("output", ""))

        provider, model_id = parse_model_option(model)
        mode, system, user_text = _resolve(prompt, task_type, width, height, skill)

        try:
            log(f"moss enhance: provider={provider} model={model_id} mode={mode} canvas={width}x{height} skill={skill} thinking={thinking}")
            log(f"user prompt: {user_text[:500]}")
            log(f"system prompt:\n{system}")
            content = chat(
                provider, model_id, system, [{"type": "text", "text": user_text}],
                temperature=temperature, seed=seed, thinking=thinking == "enabled",
                effort_choice=reasoning_effort, api_key=api_key, max_tokens=ENHANCE_MAX_TOKENS,
                on_text=make_progress_cb(cls.hidden.unique_id),
            )
            log(f"raw response ({len(content)} chars):\n{content[:1000]}")
            result = postprocess(content, mode, width, height)
            log(f"final prompt ({len(result)} chars):\n{result[:1000]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "optimize",
                "task_type": f"cloud/moss/{mode.lower()}",
                "model": f"{provider}/{model_id}",
                "thinking": thinking,
                "canvas": f"{width}x{height}",
                "input": prompt.strip(),
                "output": result,
            })
            return io.NodeOutput(result)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
