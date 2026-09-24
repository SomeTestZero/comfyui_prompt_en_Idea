import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import (
    append_history,
    find_history_entry,
    history_entry_options,
    lora_profile_options,
    log,
    make_progress_cb,
)
from .nodes_cloud import (
    ENHANCE_MAX_TOKENS,
    chat,
    image_part,
    model_combo,
    parse_model_option,
    require_vision,
)
from .nodes_qwen21 import STYLE_PRESETS, build_style_note, collect_frames
from .nodes_universal import build_lora_note
from .skills import load_skill, scan_skills

KREA2_SYSTEM = """You are a professional prompt engineer for the Krea 2 text-to-image model (RAW and Turbo). Rewrite the user's request into a final Krea-2 prompt that strictly follows the installed skill and the official prompting guide below.

Rules:
- Output ONLY the final prompt: exactly one English paragraph. No explanations, no commentary, no markdown fences.
- The attached reference images guide the description only — Krea 2 generation takes text alone. Ground style, palette, and subject details in what is visibly in them; never invent appearance details.{image_note}{style_note}{lora_note}

=== SKILL ===
{skill_body}

=== OFFICIAL KREA 2 PROMPTING GUIDE: {ref_names} ===
{ref_text}"""


def build_user_text(prompt, frames):
    """(user_text, image_note) pair; with no text request the intent is
    reverse-inferred from the attached pictures. The images ground the
    description only — Krea 2 generation is text-only."""
    user_text = prompt.strip()
    if not frames:
        return user_text, ""
    listing = ", ".join(f"<image{i + 1}>" for i in range(len(frames)))
    image_note = (
        f"\n- The user attached {len(frames)} image(s), given in the user message in order: {listing}. "
        "Describe only what is actually visible in them; never invent appearance details."
    )
    if user_text:
        user_text += f"\n\n{len(frames)} image(s) attached in order: {listing}."
    else:
        user_text = (
            f"No text request was provided. Reverse-infer the intent from the {len(frames)} attached image(s), "
            f"given in order: {listing}, and write the final Krea-2 prompt from the images alone, following the "
            "skill and guide."
        )
    return user_text, image_note


class Krea2PromptEnhancerCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = scan_skills()
        default_skill = "krea2-prompt-writing" if "krea2-prompt-writing" in skills else skills[0]
        return io.Schema(
            node_id="Krea2PromptEnhancerCloud",
            display_name="Krea 2 Prompt Enhancer (Cloud API)",
            category="prompt",
            description="Rewrite a rough request into a Krea 2 (RAW/Turbo) text-to-image prompt with a multimodal cloud model, guided by the installed skill (krea2-prompt-writing archives the official Krea 2 prompting guidelines and Krea's own expansion system prompt). Reference images ground the description only — Krea 2 generation is text-only. style_preset/style_profile pin a byte-stable style block across a set; lora_profile injects a LoRA Profiler profile and its trigger words. Supports history_entry replay across both backends.",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("style_preset", display_name="风格预设", options=["无", "自定义"] + list(STYLE_PRESETS), default="无", tooltip="钉风格块：选具名预设就把它内置的风格词逐字钉进本套图每条提示词（跨次跑措辞完全一致 → 画风统一）。自定义 = 在风格块框里写自己的风格块（英文）；无 = 不钉（但风格块框填了内容仍生效）。选了具名预设就忽略风格块框。"),
                io.String.Input("style_profile", display_name="风格块", multiline=True, default="", optional=True, tooltip="自己写的风格块（英文）：逐字钉进每条改写结果（不许被释义/改写），一套图共用同一段即画风统一。风格预设选了具名档时本框被忽略。"),
                io.Combo.Input("lora_profile", options=lora_profile_options(), default="none", optional=True, tooltip="LoRA 档案（LoRA Profiler 体检产物）：注入该 LoRA 的实测风格/角色特征，并强制最终提示词以触发词开头。列表在节点创建或刷新页面时更新。"),
                io.Combo.Input("skill", options=skills, default=default_skill, tooltip="krea2-prompt-writing = 官方 Krea 2 提示词指南 + Krea 自家扩写 system prompt（存档 + 提炼规则）。"),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True, tooltip="Deep-thinking switch (thinking.type). Doubao Seed and DeepSeek honor it; GLM models always think and ignore this - reasoning_effort controls their depth."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, tooltip="Sampling temperature; other sampling knobs follow the provider's server defaults."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Sent to the API; cloud serving reproduces outputs on a best-effort basis only."),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True, tooltip="Thinking depth. Doubao: sent as reasoning_effort (auto = server default). GLM: low/medium/high map to low/high/max (always thinking). DeepSeek ignores it."),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("image", tooltip="Reference image grounding the description; Krea 2 generation itself is text-only."),
                        prefix="image_", min=0, max=12,
                    ),
                    tooltip="参考图（可选）：只用来给描述提供风格/角色依据，Krea 2 生成本身纯文本。接图且 prompt 留空时从图反推意图。"),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored run (local or cloud) instead of calling the API. List refreshes when the node is created or the page reloads."),
                io.String.Input("api_key", default="", optional=True, tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
            ],
            outputs=[io.String.Output(display_name="enhanced_prompt")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, skill, model, thinking, temperature, seed,
                reasoning_effort="low", style_preset="无", style_profile="",
                lora_profile="none", images=None, history_entry="none", api_key=""):
        if history_entry != "none":
            e = find_history_entry(history_entry)
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            return io.NodeOutput(e.get("output", ""))

        provider, model_id = parse_model_option(model)
        frames = collect_frames(images)
        require_vision(provider, model_id, frames)
        if not prompt.strip() and not frames:
            raise ValueError("prompt is empty and no reference images connected.")

        skill_body, refs = load_skill(skill, "generic")
        user_text, image_note = build_user_text(prompt, frames)
        system = KREA2_SYSTEM.format(
            image_note=image_note,
            style_note=build_style_note(style_preset, style_profile),
            lora_note=build_lora_note(lora_profile),
            skill_body=skill_body,
            ref_names=" + ".join(n for n, _ in refs) or "none",
            ref_text="\n\n".join(t for _, t in refs),
        )
        user_content = [{"type": "text", "text": user_text}] + [image_part(f) for f in frames]

        try:
            log(f"krea2 enhance: provider={provider} model={model_id} skill={skill} images={len(frames)} thinking={thinking} effort={reasoning_effort}")
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
                "task_type": "cloud/krea2/t2i",
                "model": f"{provider}/{model_id}",
                "thinking": thinking,
                "input": prompt.strip() or f"{len(frames)} image(s), no text",
                "output": content,
            })
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
