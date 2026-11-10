# -*- coding: utf-8 -*-
"""H3 character-replacement prompt node: a brief swap instruction plus source
video frames and character references -> a full six-section H3 edit prompt.

The rewrite rules live in the h3-prompt-writing skill's characterswap-en.txt
keyed guide (loaded only for this task). Everything else reuses the cloud
node plumbing (chat / image_part / history replay)."""
import re
import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import (
    append_history,
    find_history_entry,
    history_entry_options,
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
from .swap_contract import APPEARANCE_AUTHORITY, swap_writing_guide
from .categories import PROMPT_ENHANCE

SWAP_MODE = "CharacterSwap"

SWAP_SYSTEM = """You are a professional prompt engineer for the MiniMax H3 video generation model, specialized in video character replacement edits. Rewrite the user's brief swap instruction into a final H3 prompt that strictly follows the skill and writing guide below.

Rules:
- Task mode: CharacterSwap. Follow the guide's six-section structure, section order, reference labels and subject bindings exactly.
- The attached images arrive in the user message in order: first the replacement character's reference images (<Picture 1>..), then evenly spaced sample frames of the source video (<Video 1>), each labeled with its time position.
- Write sections in English; preserve dialogue, lyrics, and visible scene text in their original language.
- Output ONLY the final prompt. No explanations, no commentary, no markdown fences.
- Before answering, self-check: the target identification (rule, code, or feature) is resolved to one unique person; all six sections present in order; <Subject 2> is defined by appearance invariants (not screen position alone); appearance details come only from the pictures and the user's words; the anti-copy clause and the temporal consistency clause are present.{image_note}

=== SKILL ===
{skill_body}

=== WRITING GUIDE: {ref_names} ===
{ref_text}"""

SIX_SECTIONS = (
    "subject_definitions:", "summary:", "retention_analysis:",
    "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
)


def sample_indices(batch, cap=6):
    """Evenly spaced indices over a source-video batch, first and last included."""
    if batch <= cap:
        return list(range(batch))
    return sorted({round(i * (batch - 1) / (cap - 1)) for i in range(cap)})


def lint_swap_prompt(text, picture_count):
    """Short actionable warnings for the two failure families: missing sections
    and dangling reference labels. Logged, never fatal - the user's value passes."""
    warns = []
    low = text.lower()
    pos = [low.find(s) for s in SIX_SECTIONS]
    if any(p < 0 for p in pos):
        warns.append("missing section(s): " + ", ".join(s for s, p in zip(SIX_SECTIONS, pos) if p < 0))
    elif pos != sorted(pos):
        warns.append("six sections present but out of order")
    pics = {int(m) for m in re.findall(r"<Picture (\d+)>", text)}
    if pics and max(pics) > picture_count:
        warns.append(f"<Picture {max(pics)}> cited but only {picture_count} reference image(s) attached")
    for tag, name in (("<Subject 1>", "replacement character"),
                      ("<Subject 2>", "replaced person"),
                      ("<Video 1>", "source video")):
        if tag not in text:
            warns.append(f"{tag} ({name}) never cited")
    return warns


def build_swap_user_text(instruction, picture_count, source_labels, duration):
    if not instruction.strip():
        raise ValueError(
            "instruction is empty: say who to replace and who replaces them, "
            "e.g. 'replace the woman in the red jacket with the character in the pictures'.")
    if not picture_count:
        raise ValueError(
            "no character reference images connected: wire them to ref_image_N "
            "(the replacement identity comes from them).")
    if not source_labels:
        raise ValueError(
            "no source video frames connected: wire the VHS_LoadVideo IMAGE output "
            "(the same frames that feed the sampler's ref_video_0) to source_frames.")
    pics = ", ".join(f"<Picture {i + 1}>" for i in range(picture_count))
    listing = "; ".join(f"<Video 1> sample {i + 1} ({lab})" for i, lab in enumerate(source_labels))
    dur_line = f"Target runtime: {duration:g} seconds." if duration > 0 else "Target runtime: not given; do not stamp shot times."
    return (
        f"{instruction.strip()}\n\n"
        f"Material given in this message, in order:\n"
        f"- {pics}: the replacement character's reference images (identity: face, hairstyle, outfit, art style).\n"
        f"- {listing}: evenly spaced sample frames of <Video 1>, the source video.\n"
        f"{dur_line}"
    )


def swap_image_note(picture_count, source_count):
    return (
        f"\n- The user attached {picture_count + source_count} image(s): the first {picture_count} are the "
        f"replacement character's reference images (<Picture 1>..), the remaining {source_count} are sample "
        "frames of the source video (<Video 1>). Describe only what is actually visible; never invent "
        "appearance details."
    )


def build_target_check(text, picture_count):
    """Pre-run confirmation block: the cast mapping (every <Subject N> definition
    line) plus lint warnings, so a mis-anchored target is caught before a full
    generation - codes and rule-based identification resolve here."""
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("<Subject")]
    head = "\n".join(lines) if lines else "(no <Subject N> definitions found - anchoring unknown)"
    warns = lint_swap_prompt(text, picture_count)
    tail = "lint: " + ("; ".join(warns) if warns else "OK")
    return f"{head}\n{tail}"


class H3SwapPromptCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SwapPromptCloud",
            display_name="[yyp] H3 Swap Prompt (Cloud API) 人物替换提示词",
            category=PROMPT_ENHANCE,
            description="Write a MiniMax H3 video character-replacement prompt from a one-line swap instruction: the "
                        "cloud model reads the source video sample frames and the character reference images, and emits "
                        "the full six-section edit prompt (subject bindings, anti-copy and temporal-consistency clauses, "
                        "per-shot timeline). Wire the output straight to MiniMax H3 Reference to Video's prompt. Rewrite "
                        "rules live in the h3-prompt-writing skill's characterswap guide; ref_image_N socket names mirror "
                        "the Reference to Video node. Supports history_entry replay.",
            inputs=[
                io.String.Input("instruction", multiline=True, default="", tooltip="一句话替换指令，三种指认可混用：规则式（'替换左1'/'the only man'）、编码式（'用 <Picture 1> 替换 P1，P2 保留'）、特征式（'红夹克女人'）。LLM 看采样帧建角色表把指认解析到唯一的人，位置变了靠外观不变量+跟踪条款锁身份。名字/触发词逐字写，不会被翻译。"),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("ref_image", tooltip="替换角色参考图；ref_image_N 与 MiniMax H3 Reference to Video 同名同号，接同一张图。"),
                        prefix="ref_image_", min=0, max=9,
                    ),
                    tooltip="替换角色参考图（1-3 张互补视角最佳）；socket 名与 MiniMax H3 Reference to Video 一致，同一张图接同号。",
                ),
                io.Image.Input("source_frames", optional=True, tooltip="源视频帧批：接 VHS_LoadVideo 的 IMAGE 输出（与喂给 ref_video_0 的同一路），节点内部均匀采样 6 帧给 LLM 看——它据此指认被换人、写动作与镜头。"),
                model_combo(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True, tooltip="Deep-thinking switch (thinking.type). Doubao Seed and DeepSeek honor it; GLM models always think and ignore this - reasoning_effort controls their depth."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, tooltip="Sampling temperature; other sampling knobs follow the provider's server defaults."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Sent to the API; cloud serving reproduces outputs on a best-effort basis only."),
                io.Combo.Input("reasoning_effort", options=["auto", "low", "medium", "high"], default="low", advanced=True, tooltip="Thinking depth. Doubao: sent as reasoning_effort (auto = server default). GLM: low/medium/high map to low/high/max (always thinking). DeepSeek ignores it."),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored run instead of calling the API. List refreshes when the node is created or the page reloads."),
                io.String.Input("api_key", default="", optional=True, tooltip="API key for the selected provider, overriding config.json and the environment variable for this run."),
                # Appended last: saved workflows store widget values positionally.
                io.Float.Input("duration", default=0.0, min=0.0, max=300.0, step=0.5, tooltip="成片时长（秒）：接采样同款 duration，用于 [Shot N] 时间戳与节奏标定。0 = 不标时间戳。"),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="target_check", tooltip="锚定确认：角色表（全部 <Subject N> 定义行）+ lint 结果。接 ShowText 先看“到底换谁”（含 P 编码/规则指认的解析结果）再放行生成；不对改 instruction 重抽。"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, instruction, model, thinking, temperature, seed,
                reasoning_effort="low", ref_images=None, source_frames=None,
                history_entry="none", api_key="", duration=0.0):
        if history_entry != "none":
            e = find_history_entry(history_entry)
            if e is None:
                raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
            log(f"history replay: {history_entry}")
            out = e.get("output", "")
            return io.NodeOutput(out, build_target_check(out, 0))

        provider, model_id = parse_model_option(model)
        picture_count = sum(len(v) for v in (ref_images or {}).values())
        if source_frames is not None and source_frames.ndim == 3:
            source_frames = source_frames.unsqueeze(0)
        batch = 0 if source_frames is None else source_frames.shape[0]
        labels = []
        if batch:
            for i in sample_indices(batch):
                if duration > 0:
                    labels.append(f"t={i / max(batch - 1, 1) * duration:.2f}s")
                else:
                    labels.append(f"frame {i}")
        user_text = build_swap_user_text(instruction, picture_count, labels, duration)

        pairs = []
        for name in sorted(ref_images or {}, key=lambda n: int(n.rsplit("_", 1)[-1])):
            for frame in ref_images[name]:
                pairs.append((frame, "replacement character reference"))
        for i, lab in zip(sample_indices(batch), labels):
            pairs.append((source_frames[i], f"source video sample at {lab}"))
        require_vision(provider, model_id, pairs)

        skill_body, refs = swap_writing_guide()
        system = SWAP_SYSTEM.format(
            image_note=swap_image_note(picture_count, len(labels)),
            skill_body=skill_body,
            ref_names=" + ".join(n for n, _ in refs) or "none",
            ref_text="\n\n".join(t for _, t in refs),
        )
        system += '\n\n=== APPEARANCE AUTHORITY ===\n' + APPEARANCE_AUTHORITY
        user_content = [{"type": "text", "text": user_text}] + [image_part(f) for f, _ in pairs]

        try:
            log(f"cloud swap: provider={provider} model={model_id} pictures={picture_count} source_frames={batch}(sampled {len(labels)}) duration={duration:g}s thinking={thinking} effort={reasoning_effort}")
            log(f"user prompt: {user_text[:500]}")
            content = chat(
                provider, model_id, system, user_content,
                temperature=temperature, seed=seed, thinking=thinking == "enabled",
                effort_choice=reasoning_effort, api_key=api_key, max_tokens=ENHANCE_MAX_TOKENS,
                on_text=make_progress_cb(cls.hidden.unique_id),
            )
            for w in lint_swap_prompt(content, picture_count):
                log(f"swap prompt warning: {w}")
            target_check = build_target_check(content, picture_count)
            log(f"response ({len(content)} chars):\n{content[:1000]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "swap",
                "task_type": f"cloud/{SWAP_MODE.lower()}",
                "model": f"{provider}/{model_id}",
                "thinking": thinking,
                "input": instruction.strip(),
                "output": content,
            })
            return io.NodeOutput(content, target_check)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
