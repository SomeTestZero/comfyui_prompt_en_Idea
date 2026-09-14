import os
import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import (
    append_history,
    call_chat_completions,
    clean_api_key,
    find_history_entry,
    history_entry_options,
    load_config,
    log,
    make_progress_cb,
)
from .llm_local import tensor_to_base64_jpeg
from .nodes_api import TRANSLATE_SYSTEM
from .nodes_idea import (
    IDEA_SYSTEM,
    MODE_GUIDANCE,
    build_idea_instruction,
    build_segment_note,
    collect_idea_frames,
)
from .nodes_local import (
    SYSTEM_TEMPLATE,
    build_enhance_user_text,
    collect_frames,
    derive_mode,
)
from .skills import load_skill, scan_skills

# OpenAI-compatible cloud endpoints. config.json's "cloud" section overrides
# the built-in base_url / api_key per provider and can add custom providers
# (any key with a base_url works, e.g. an OpenAI-compatible relay).
PROVIDERS = {
    "volcengine": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "env_key": "VOLCENGINE_API_KEY",
    },
    # Agent/Coding Plan setmeal endpoints: one key, shared monthly quota, the
    # same OpenAI-compatible surface under /api/plan/v3 and /api/coding/v3
    # (mirrors the pi-volcengine-plans provider split).
    "volcengine-plan": {
        "base_url": "https://ark.cn-beijing.volces.com/api/plan/v3",
        "env_key": "VOLCENGINE_AGENT_PLAN_API_KEY",
    },
    "volcengine-coding": {
        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "env_key": "VOLCENGINE_CODING_PLAN_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
    },
}

# Manually maintained model lists, pi-volcengine-plans style: no remote
# fetching, offline-safe. Add a model by copying one line and editing it.
#   vision: accepts image_url content blocks; unknown models (added via
#           config.json) run unprofiled - images allowed, no effort mapping.
#   thinking: "controlled" honors the thinking input (thinking.type parameter),
#             "forced" always thinks server-side and ignores it (GLM); also
#             used for models whose switch is unverified - the field is then
#             simply not sent (Kimi/MiniMax on the plan endpoint).
#   effort: maps the reasoning_effort input to the API value; missing key or
#           absent field = parameter not sent (auto = server default).
MODELS = {
    "volcengine": [
        # Doubao Seed 2.1 (2026-06-23 release): vision + thinking switch + effort.
        {"id": "doubao-seed-2-1-pro-260628", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        {"id": "doubao-seed-2-1-turbo-260628", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        # Rolling flagship, one stable id upgraded weekly.
        {"id": "doubao-seed-evolving", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        {"id": "doubao-seed-2-0-pro-260215", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
    ],
    "volcengine-plan": [
        # Agent Plan endpoint, verified against a real plan key on 2026-09-13:
        # a color-tracking test (red/green image, answer must follow) separates
        # true vision models from ones that silently ignore image blocks.
        {"id": "glm-5.3-flash", "vision": True, "thinking": "forced",
         "effort": {"low": "low", "medium": "high", "high": "max"}},
        {"id": "doubao-seed-2-1-turbo-260628", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        {"id": "doubao-seed-evolving", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        {"id": "doubao-seed-2.0-pro", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        {"id": "doubao-seed-2-0-pro-260215", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        {"id": "doubao-seed-2.0-lite", "vision": True, "thinking": "controlled",
         "effort": {"low": "low", "medium": "medium", "high": "high"}},
        {"id": "kimi-k3", "vision": True, "thinking": "forced"},
        {"id": "kimi-k2.7-code", "vision": True, "thinking": "forced"},
        # text-only: image blocks are rejected with a clear 400 by the GLM trio
        {"id": "glm-5.3", "vision": False, "thinking": "forced",
         "effort": {"low": "low", "medium": "high", "high": "max"}},
        {"id": "glm-5.2", "vision": False, "thinking": "forced"},
        {"id": "glm-5.3v", "vision": False, "thinking": "forced"},
        # silently ignore image blocks (answer without looking) - marked
        # vision=False so the nodes reject image input up front
        {"id": "deepseek-v4-flash", "vision": False, "thinking": "controlled"},
        {"id": "deepseek-v4-pro", "vision": False, "thinking": "controlled"},
        {"id": "deepseek-v4-flash-vision-exp", "vision": False, "thinking": "controlled"},
        {"id": "minimax-m3", "vision": False, "thinking": "forced"},
    ],
    "volcengine-coding": [
        {"id": "glm-5.3-flash", "vision": True, "thinking": "forced",
         "effort": {"low": "low", "medium": "high", "high": "max"}},
        {"id": "glm-5.3", "vision": False, "thinking": "forced",
         "effort": {"low": "low", "medium": "high", "high": "max"}},
    ],
    "deepseek": [
        # V4-Flash-Vision-Exp (2026-08-21): DeepSeek's first image-input model.
        {"id": "deepseek-v4-flash-vision-exp", "vision": True, "thinking": "controlled"},
        {"id": "deepseek-v4-flash", "vision": False, "thinking": "controlled"},
        {"id": "deepseek-v4-pro", "vision": False, "thinking": "controlled"},
    ],
}

DEFAULT_MODEL = "volcengine-plan/glm-5.3-flash"

IDEA_MAX_TOKENS = 16384
ENHANCE_MAX_TOKENS = 16384
TRANSLATE_MAX_TOKENS = 8192


def cloud_model_options():
    """'provider/model' combo options: built-in lists plus models appended via
    config.json cloud.<provider>.models (and whole custom providers)."""
    cloud_cfg = load_config().get("cloud", {})
    options = []
    for provider, models in MODELS.items():
        ids = [m["id"] for m in models]
        options += [f"{provider}/{mid}" for mid in ids]
        options += [f"{provider}/{mid}" for mid in cloud_cfg.get(provider, {}).get("models", []) if mid not in ids]
    for provider, pcfg in cloud_cfg.items():
        if provider not in MODELS:
            options += [f"{provider}/{mid}" for mid in pcfg.get("models", [])]
    return options


def parse_model_option(option):
    provider, sep, model = option.partition("/")
    if not sep:
        raise ValueError(f"Unknown cloud model: {option!r} - pick one from the model dropdown.")
    return provider, model


def model_info(provider, model):
    """Built-in metadata for a model id, or None for unprofiled config-added models."""
    for m in MODELS.get(provider, []):
        if m["id"] == model:
            return m
    return None


def resolve_provider(provider):
    """(base_url, api_key from config) - built-in defaults overridable per provider."""
    pcfg = load_config().get("cloud", {}).get(provider, {})
    base_url = str(pcfg.get("base_url") or PROVIDERS.get(provider, {}).get("base_url", "")).rstrip("/")
    if not base_url:
        raise ValueError(f"cloud provider '{provider}' has no base_url - set cloud.{provider}.base_url in config.json.")
    return base_url, str(pcfg.get("api_key", "")).strip()


def resolve_api_key(provider, node_key):
    """Node input > config.json cloud.<provider>.api_key > provider env var.
    The deepseek provider also falls back to the legacy top-level api_key."""
    key = clean_api_key(node_key or "")
    if not key:
        key = resolve_provider(provider)[1]
    if not key and provider == "deepseek":
        key = str(load_config().get("api_key", "")).strip()
    if not key:
        key = os.environ.get(PROVIDERS.get(provider, {}).get("env_key", ""), "").strip()
    if not key:
        env = PROVIDERS.get(provider, {}).get("env_key")
        env_note = f", the {env} environment variable" if env else ""
        raise ValueError(f"API key for cloud provider '{provider}' is empty. Set it via the api_key input{env_note}, or cloud.{provider}.api_key in config.json.")
    return key


def require_vision(provider, model, frames):
    """Clear error instead of a server-side rejection when images meet a text model."""
    info = model_info(provider, model)
    if frames and info is not None and not info["vision"]:
        raise ValueError(f"{provider}/{model} does not accept image input - pick a model with vision support or disconnect the images.")


def image_part(frame):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{tensor_to_base64_jpeg(frame)}"},
    }


def chat(provider, model, system, user_content, *, temperature, seed, thinking,
         effort_choice, api_key, max_tokens, on_text=None):
    base_url, _ = resolve_provider(provider)
    key = resolve_api_key(provider, api_key)
    info = model_info(provider, model) or {}
    controlled = info.get("thinking", "controlled") == "controlled"
    effort = (info.get("effort") or {}).get(effort_choice)
    # Ark 400s on reasoning_effort combined with thinking disabled; with
    # thinking off there is no depth to tune anyway.
    if controlled and not thinking:
        effort = None
    return call_chat_completions(
        {"base_url": base_url}, key, model, system, user_content, temperature,
        on_text=on_text,
        thinking=thinking,
        seed=seed,
        max_tokens=max_tokens,
        control_thinking=controlled,
        reasoning_effort=effort,
    )


def replay_entry_mode(entry, task_type):
    """Mode of a stored history record, local or cloud (T2VA fallback)."""
    stored = str(entry.get("task_type", ""))
    for prefix in ("local/", "cloud/"):
        if stored.startswith(prefix):
            return stored[len(prefix):]
    return "T2VA" if task_type == "Auto" else task_type


def model_combo():
    return io.Combo.Input(
        "model", options=cloud_model_options(), default=DEFAULT_MODEL,
        tooltip="provider/model pairs on OpenAI-compatible cloud endpoints. "
                "volcengine = pay-as-you-go Ark, volcengine-plan/coding = Agent/Coding Plan setmeal keys, "
                "deepseek = DeepSeek API. Image inputs need a vision model (Doubao Seed, GLM-5.3-Flash, "
                "DeepSeek V4-Flash-Vision). Extra models: cloud.<provider>.models in config.json.",
    )


class H3IdeaGeneratorCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3IdeaGeneratorCloud",
            display_name="H3 Idea Generator (Cloud API) 灵感生成",
            category="prompt",
            description="Cloud API variant of the H3 Idea Generator: the same seed-drawn story drafts (setup + shot-by-shot beats, duration-stamped, long-take beat sheets), written by a multimodal cloud model instead of a local GGUF. Image inputs need a vision model; the seed draws the same ingredients as the local node, so both backends riff on the same reproducible combinations. API key: api_key input > config.json cloud.<provider>.api_key > provider environment variable.",
            inputs=[
                io.String.Input("hint", multiline=True, default="", tooltip="Optional theme/clue to riff on - highest priority, everything it does not specify is filled from seed-picked ingredients, so a style-only hint like '3DCG' still gets a different subject/setting/event every seed (conflicting ingredients are dropped). No need to write the duration here, it has its own input."),
                io.Combo.Input("language", options=["中文", "English"], default="中文", tooltip="Language of the generated story; the enhancer accepts either."),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True, tooltip="Deep-thinking switch (thinking.type). Doubao Seed and DeepSeek honor it; GLM models always think and ignore this - reasoning_effort controls their depth."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, tooltip="Sampling temperature; other sampling knobs follow the provider's server defaults."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Drives the local ingredient draw exactly like the local node; also sent to the API, where cloud serving reproduces outputs on a best-effort basis only."),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True, tooltip="Thinking depth. Doubao: sent as reasoning_effort (auto = server default). GLM: low/medium/high map to low/high/max (always thinking). DeepSeek ignores it."),
                io.Image.Input("first_frame", optional=True, tooltip="First keyframe image; the story starts from this scene (with last_frame: ends at that scene). Same wiring as the enhancer."),
                io.Image.Input("last_frame", optional=True, tooltip="Last keyframe image; the story builds up to this scene."),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("ref_image", tooltip="Subject/character reference; the story stars what is visible in it. ref_image_N matches the enhancer's socket of the same name."),
                        prefix="ref_image_", min=0, max=9,
                    ),
                    tooltip="Reference images (characters, style, props); socket names mirror the H3 Prompt Enhancer / MiniMax H3 Reference to Video nodes — wire the same image to the same-numbered ref_image_N on all of them.",
                ),
                # Appended last: saved workflows store widget values positionally.
                io.Float.Input("duration", default=10.0, min=4.0, max=300.0, step=0.5, tooltip="Target video duration in seconds - wire the same duration that feeds the sampler's frame count. The story is sized to fit it, and the value is stamped on the draft so the enhancer schedules shots within the runtime."),
                io.Float.Input("segment_seconds", default=0.0, min=0.0, max=15.0, step=0.5, tooltip="Long-take chaining: wire the planner's segment_seconds. 0 = single-clip draft (default)."),
                io.Int.Input("total_segments", default=0, min=0, max=99, tooltip="Long-take chaining: wire the planner's total_segments so the draft gets exactly this many numbered beats. 0/1 = single-clip draft."),
                io.String.Input("beat_durations", default="", optional=True, tooltip="Long-take chaining: wire the planner's segment_durations (comma-separated story seconds per segment, e.g. \"8.0, 7.1, 7.1, 7.1, 0.7\") so each beat is sized to its real story time."),
                io.String.Input("api_key", default="", optional=True, tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
            ],
            outputs=[io.String.Output(display_name="idea")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, hint, language, model, thinking, temperature, seed,
                reasoning_effort="low", first_frame=None, last_frame=None,
                reference_images=None, duration=10.0,
                segment_seconds=0.0, total_segments=0, beat_durations="", api_key=""):
        provider, model_id = parse_model_option(model)
        mode = derive_mode(first_frame, last_frame, reference_images)
        dur = f"{duration:g}"

        frames = collect_idea_frames(first_frame, last_frame, reference_images)
        require_vision(provider, model_id, frames)
        user_text = build_idea_instruction(hint, frames, seed)

        image_note = f"\n- {MODE_GUIDANCE[mode]}" if frames else ""
        segment_note = build_segment_note(segment_seconds, total_segments, beat_durations)
        system = IDEA_SYSTEM.format(image_note=image_note, segment_note=segment_note, language=language, duration=dur)
        user_content = [{"type": "text", "text": user_text}] + [image_part(f) for f, _ in frames]

        try:
            log(f"cloud idea: provider={provider} model={model_id} mode={mode} seed={seed} lang={language} duration={dur}s images={len(frames)} segments={total_segments}x{segment_seconds:g}s thinking={thinking} hint={hint.strip()[:100]!r}")
            log(f"user prompt: {user_text[:500]}")
            content = chat(
                provider, model_id, system, user_content,
                temperature=temperature, seed=seed, thinking=thinking == "enabled",
                effort_choice=reasoning_effort, api_key=api_key, max_tokens=IDEA_MAX_TOKENS,
                on_text=make_progress_cb(cls.hidden.unique_id),
            )
            # Stamp the runtime on the draft so the downstream enhancer can
            # schedule shot timings against the requested duration.
            header = f"总时长：{dur} 秒" if language == "中文" else f"Total runtime: {dur} seconds"
            content = f"{header}\n\n{content}"
            log(f"idea ({len(content)} chars):\n{content[:500]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "idea",
                "task_type": f"cloud/idea/{mode}",
                "model": f"{provider}/{model_id}",
                "seed": seed,
                "duration": duration,
                "total_segments": total_segments,
                "input": hint.strip() or (f"random, {len(frames)} image(s)" if frames else "random"),
                "output": content,
            })
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise


class H3PromptEnhancerCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3PromptEnhancerCloud",
            display_name="H3 Prompt Enhancer (Cloud API)",
            category="prompt",
            description="Cloud API variant of the H3 Prompt Enhancer: rewrites a rough request into a structured MiniMax H3 prompt with a multimodal cloud model, guided by the same installed skill and image wiring as the local node (reference images, keyframes, empty-prompt reverse inference). Needs a vision model when images are connected. Supports history_entry replay across both backends.",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("task_type", options=["Auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"], default="Auto", tooltip="Auto: derived from which image sockets are connected — reference images -> Ref2VA, first+last frame -> FL2VA, first only -> I2VA, last only -> L2VA, none -> T2VA."),
                io.Combo.Input("skill", options=scan_skills(), default="h3-prompt-writing" if "h3-prompt-writing" in scan_skills() else scan_skills()[0]),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True, tooltip="Deep-thinking switch (thinking.type). Doubao Seed and DeepSeek honor it; GLM models always think and ignore this - reasoning_effort controls their depth."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, tooltip="Sampling temperature; other sampling knobs follow the provider's server defaults."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Sent to the API; cloud serving reproduces outputs on a best-effort basis only."),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True, tooltip="Thinking depth. Doubao: sent as reasoning_effort (auto = server default). GLM: low/medium/high map to low/high/max (always thinking). DeepSeek ignores it."),
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
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored run (local or cloud) instead of calling the API. List refreshes when the node is created or the page reloads."),
                io.String.Input("api_key", default="", optional=True, tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
            ],
            outputs=[
                io.String.Output(display_name="enhanced_prompt"),
                io.String.Output(display_name="detected_mode"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, task_type, skill, model, thinking, temperature, seed,
                reasoning_effort="low", first_frame=None, last_frame=None,
                reference_images=None, history_entry="none", api_key=""):
        if history_entry != "none":
            e = find_history_entry(history_entry)
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(e.get("output", ""), replay_entry_mode(e, task_type))

        provider, model_id = parse_model_option(model)
        mode = derive_mode(first_frame, last_frame, reference_images) if task_type == "Auto" else task_type
        frames = collect_frames(mode, first_frame, last_frame, reference_images)
        require_vision(provider, model_id, frames)
        if not prompt.strip() and not frames:
            raise ValueError("prompt is empty and no reference images connected.")

        skill_body, refs = load_skill(skill, mode)
        user_text, image_note = build_enhance_user_text(prompt, frames)
        system = SYSTEM_TEMPLATE.format(
            mode=mode,
            image_note=image_note,
            skill_body=skill_body,
            ref_names=" + ".join(n for n, _ in refs) or "none",
            ref_text="\n\n".join(t for _, t in refs),
        )
        user_content = [{"type": "text", "text": user_text}] + [image_part(f) for f, _ in frames]

        try:
            log(f"cloud enhance: provider={provider} model={model_id} mode={mode} skill={skill} images={len(frames)} thinking={thinking} effort={reasoning_effort}")
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
                "task_type": f"cloud/{mode}",
                "model": f"{provider}/{model_id}",
                "thinking": thinking,
                "input": prompt.strip() or f"{len(frames)} image(s), no text",
                "output": content,
            })
            return io.NodeOutput(content, mode)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise


class H3TranslatorCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3TranslatorCloud",
            display_name="H3 Translator (Cloud API) 审核翻译",
            category="prompt",
            description="Cloud API variant of the H3 Translator: same review-oriented translation rules (structure, shot markers, reference labels and timing notation preserved), executed by a cloud model. Text-only input - any model in the list works.",
            inputs=[
                io.String.Input("text", multiline=True, default="", tooltip="Linked input hides the box; widget kept (not force_input) so saved workflows round-trip correctly."),
                io.Combo.Input("target_lang", options=["中文", "English"], default="中文"),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True, tooltip="Deep-thinking switch (thinking.type). Doubao Seed and DeepSeek honor it; GLM models always think and ignore this - reasoning_effort controls their depth."),
                io.Float.Input("temperature", default=0.3, min=0.0, max=2.0, step=0.05),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True, tooltip="Thinking depth. Doubao: sent as reasoning_effort (auto = server default). GLM: low/medium/high map to low/high/max (always thinking). DeepSeek ignores it."),
                io.String.Input("api_key", default="", optional=True, tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
            ],
            outputs=[io.String.Output(display_name="translated_text")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, text, target_lang, model, thinking, temperature,
                reasoning_effort="low", api_key=""):
        if not text.strip():
            log("cloud translate: empty input, passing through")
            return io.NodeOutput("")
        provider, model_id = parse_model_option(model)
        system = TRANSLATE_SYSTEM.format(target=target_lang)
        try:
            log(f"cloud translate: provider={provider} model={model_id} target={target_lang}")
            log(f"input text: {text.strip()[:500]}")
            content = chat(
                provider, model_id, system, text.strip(),
                temperature=temperature, seed=None, thinking=thinking == "enabled",
                effort_choice=reasoning_effort, api_key=api_key, max_tokens=TRANSLATE_MAX_TOKENS,
                on_text=make_progress_cb(cls.hidden.unique_id),
            )
            log(f"translation ({len(content)} chars):\n{content[:1000]}")
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
