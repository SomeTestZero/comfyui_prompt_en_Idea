import os
import traceback
from datetime import datetime

from comfy_api.latest import io

import folder_paths

from .common import (
    _last_generation,
    append_history,
    call_chat_completions,
    call_responses_api,
    clean_api_key,
    fetch_models,
    find_history_entry,
    history_entry_options,
    load_config,
    log,
    make_progress_cb,
)

SYSTEM_TEMPLATE = """You are a professional prompt engineer for AI generation models (video, image, etc.). Rewrite the user's rough idea into a final prompt that strictly follows the official writing guide below.

Rules:
- Follow the guide's structure and formatting exactly.
- Write the optimized prompt in English; keep dialogue, lyrics and on-screen text in their original language.
- Output ONLY the final optimized prompt. No explanations, no commentary.
{task_hint}
=== OFFICIAL GUIDE ===
{guide}"""


GENERIC_OPTIMIZE_SYSTEM = """You are a professional prompt engineer for AI generation models (video, image, etc.). Rewrite the user's rough idea into a detailed, well-structured prompt.

Rules:
- Write the optimized prompt in English; keep dialogue, lyrics and on-screen text in their original language.
- Output ONLY the final optimized prompt. No explanations, no commentary."""


TASK_TYPE_OPTIONS = [
    "Auto - 自动判断",
    "T2VA - 纯文本生成 (无参考图)",
    "I2VA - 首帧参考图",
    "FL2VA - 首尾帧参考图",
    "L2VA - 末帧参考图",
    "reference_generation - 角色/风格参考",
]

H3_SKILL_DOC = os.path.join("minimax_h3", "skills", "h3-prompt-writing", "SKILL.md")
H3_BASE_GUIDE = os.path.join("minimax_h3", "VIDEO_PROMPT_WRITING_GUIDE_base_en.md")
H3_REF_GUIDE = os.path.join("minimax_h3", "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md")


def resolve_auto_refs(task_type):
    """Official h3-prompt-writing skill routing: guide selection follows the task mode."""
    if task_type.startswith("reference_generation"):
        return [H3_SKILL_DOC, H3_REF_GUIDE]
    if task_type.startswith("Auto"):
        return [H3_SKILL_DOC, H3_BASE_GUIDE, H3_REF_GUIDE]
    return [H3_SKILL_DOC, H3_BASE_GUIDE]


TASK_TYPE_HINTS = {
    "Auto - 自动判断": "",
    "T2VA - 纯文本生成 (无参考图)": "- Task type: T2VA. Follow the corresponding section of the guide.",
    "I2VA - 首帧参考图": "- Task type: I2VA. Follow the corresponding section of the guide.",
    "FL2VA - 首尾帧参考图": "- Task type: FL2VA. Follow the corresponding section of the guide.",
    "L2VA - 末帧参考图": "- Task type: L2VA. Follow the corresponding section of the guide.",
    "reference_generation - 角色/风格参考": "- Task type: reference_generation. Output the six full-reference sections in order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music, as defined in the Full-Reference Mode guide. IMPORTANT: You cannot see the reference images, so NEVER invent a subject's appearance, gender, clothing, or identity. In subject_definitions, only map each <Subject N> to its source, e.g. '<Subject 1> is the fighter whose appearance comes from <Picture 1>.' Do not create reference subjects for freely invented content such as the environment, props, or effects; describe those as ordinary setting. In the description, use the labels to refer to the referenced characters without adding any fabricated appearance details.",
}


def load_reference_docs(root, entries):
    """Load reference docs. An entry ending in a path separator means the whole folder."""
    doc_names, doc_texts = [], []
    for entry in dict.fromkeys(entries):
        if entry == "none":
            continue
        path = os.path.abspath(os.path.join(root, entry))
        if not path.startswith(root + os.sep):
            raise FileNotFoundError(f"Reference path not found under {root}: {entry}")
        if entry.endswith(("/", os.sep)):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Reference folder not found under {root}: {entry}")
            for dirpath, _, names in os.walk(path):
                for name in sorted(names):
                    if not name.lower().endswith((".md", ".txt")):
                        continue
                    with open(os.path.join(dirpath, name), "r", encoding="utf-8") as f:
                        doc_names.append(os.path.relpath(os.path.join(dirpath, name), root))
                        doc_texts.append(f.read())
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Reference file not found under {root}: {entry}")
            with open(path, "r", encoding="utf-8") as f:
                doc_names.append(entry)
                doc_texts.append(f.read())
    return doc_names, doc_texts


class DeepSeekPromptOptimizer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DeepSeekPromptOptimizer",
            display_name="DeepSeek Prompt Optimizer",
            category="prompt",
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("mode", options=["prompt_optimize", "chat"], default="prompt_optimize"),
                io.Combo.Input("task_type", options=TASK_TYPE_OPTIONS, default=TASK_TYPE_OPTIONS[0]),
                io.Combo.Input("model", options=fetch_models()),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled"),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Boolean.Input("web_search", default=False),
                io.Boolean.Input("keep_previous_output", default=False),
                io.Combo.Input("history_entry", options=history_entry_options(), default="none", optional=True, tooltip="Replay a stored record instead of calling the API. List refreshes when the node is created or the page reloads."),
                io.String.Input("api_key", default="", optional=True),
            ],
            outputs=[io.String.Output(display_name="optimized_prompt")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, prompt, mode, task_type, model, thinking, temperature, web_search,
                keep_previous_output, history_entry="none", api_key=""):
        try:
            if history_entry != "none":
                e = find_history_entry(history_entry)
                if e is None:
                    raise ValueError(f"History entry not found (list may be stale, reselect it): {history_entry}")
                log(f"history replay: {history_entry}")
                return io.NodeOutput(e.get("output", ""))

            cache_key = (prompt, mode, task_type, model, thinking, temperature, web_search)
            if keep_previous_output and _last_generation["output"] and _last_generation["key"] == cache_key:
                log("keep_previous_output: returning cached output")
                return io.NodeOutput(_last_generation["output"])

            cfg = load_config()
            key = clean_api_key(api_key) or cfg["api_key"].strip()
            if not key:
                raise ValueError(f"DeepSeek API key is empty. Set it in config.json or in the node's api_key input.")

            if not prompt.strip():
                raise ValueError("prompt is empty.")

            root = os.path.abspath(os.path.join(folder_paths.base_path, "model_doc"))
            doc_names, doc_texts = load_reference_docs(root, resolve_auto_refs(task_type))
            ref_text = "\n\n".join(doc_texts)

            if mode == "prompt_optimize":
                if ref_text:
                    task_hint = TASK_TYPE_HINTS.get(task_type, "")
                    system = SYSTEM_TEMPLATE.format(task_hint=task_hint, guide=ref_text)
                else:
                    system = GENERIC_OPTIMIZE_SYSTEM
            else:
                system = "You are a helpful assistant. Answer the user's question directly and concisely."
                if ref_text:
                    system += f"\n\nAnswer using the reference document below when it is relevant.\n=== REFERENCE DOCUMENT ({' + '.join(doc_names)}) ===\n{ref_text}"

            log(f"mode={mode} task_type={task_type} model={model} thinking={thinking} web_search={web_search}")
            log(f"user prompt: {prompt.strip()[:500]}")
            log(f"system prompt:\n{system}")

            if web_search:
                if model != "deepseek-v4-flash":
                    raise ValueError(f"web_search is only available with the deepseek-v4-flash model, current model: {model}")
                content = call_responses_api(cfg, key, model, system, prompt.strip(), temperature)
            else:
                content = call_chat_completions(cfg, key, model, system, prompt.strip(), temperature, on_text=make_progress_cb(cls.hidden.unique_id), thinking=(thinking == "enabled"))
            log(f"response ({len(content)} chars):\n{content[:1000]}")
            _last_generation["key"] = cache_key
            _last_generation["output"] = content
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "optimize",
                "task_type": task_type,
                "model": model,
                "thinking": thinking,
                "input": prompt.strip(),
                "output": content,
            })
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise


TRANSLATE_SYSTEM = """You are a professional translator for AI video generation prompts. Translate the user's text into {target}.

Rules:
- Preserve all structure and formatting exactly: section names (e.g. integrated_multimodal_description, overall_soundscape), shot markers (e.g. [Shot 1]), reference labels (e.g. <Subject 1>, <Picture 2>), and timing notation stay untranslated so the translation can be checked line-by-line against the original.
- Translate only descriptive prose. Keep dialogue and lyrics in their original language, adding the {target} meaning in parentheses when helpful.
- Output ONLY the translation. No explanations, no commentary."""


class DeepSeekTranslator(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DeepSeekTranslator",
            display_name="DeepSeek Translator (审核翻译)",
            category="prompt",
            inputs=[
                io.String.Input("text", multiline=True, default=""),
                io.Combo.Input("target_lang", options=["中文", "English"], default="中文"),
                io.Combo.Input("model", options=fetch_models()),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled"),
                io.Float.Input("temperature", default=0.3, min=0.0, max=2.0, step=0.05),
                io.String.Input("api_key", default="", optional=True),
            ],
            outputs=[io.String.Output(display_name="translated_text")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, text, target_lang, model, thinking, temperature, api_key=""):
        try:
            if not text.strip():
                log("translate: empty input, passing through")
                return io.NodeOutput("")
            cfg = load_config()
            key = clean_api_key(api_key) or cfg["api_key"].strip()
            if not key:
                raise ValueError(f"DeepSeek API key is empty. Set it in config.json or in the node's api_key input.")
            system = TRANSLATE_SYSTEM.format(target=target_lang)
            log(f"translate target={target_lang} model={model}")
            log(f"input text: {text.strip()[:500]}")
            content = call_chat_completions(cfg, key, model, system, text.strip(), temperature, on_text=make_progress_cb(cls.hidden.unique_id), thinking=(thinking == "enabled"))
            log(f"translation ({len(content)} chars):\n{content[:1000]}")
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
