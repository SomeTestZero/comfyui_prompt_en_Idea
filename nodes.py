import json
import os
import time

import requests

import folder_paths

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(NODE_DIR, "config.json")
DOC_DIR = os.path.join(folder_paths.base_path, "model_doc", "minimax_h3")

GUIDE_FILES = {
    "base": "VIDEO_PROMPT_WRITING_GUIDE_base_en.md",
    "ref": "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md",
}


def load_config():
    cfg = {"api_key": "", "base_url": "https://api.deepseek.com"}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


_model_list_cache = {"models": [], "fetched_at": 0.0}
MODEL_LIST_TTL = 3600


def fetch_models():
    now = time.time()
    if _model_list_cache["models"] and now - _model_list_cache["fetched_at"] < MODEL_LIST_TTL:
        return _model_list_cache["models"]
    cfg = load_config()
    if cfg["api_key"].strip():
        try:
            r = requests.get(
                cfg["base_url"].rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {cfg['api_key'].strip()}"},
                timeout=10,
            )
            if r.status_code == 200:
                models = sorted(m["id"] for m in r.json().get("data", []) if "id" in m)
                if models:
                    _model_list_cache["models"] = models
                    _model_list_cache["fetched_at"] = now
        except requests.RequestException:
            pass
    return _model_list_cache["models"] or ["deepseek-chat", "deepseek-reasoner"]


def load_guides(choice):
    names = list(GUIDE_FILES.values()) if choice == "base+ref" else [GUIDE_FILES[choice]]
    parts = []
    for name in names:
        path = os.path.join(DOC_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Guide document not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n\n".join(parts)


SYSTEM_TEMPLATE = """You are a professional prompt engineer for the MiniMax Hailuo video generation model. Rewrite the user's rough idea into a final video prompt that strictly follows the official writing guide below.

Rules:
- Follow the guide's structure and formatting exactly.
- Write the optimized prompt in English; keep dialogue, lyrics and on-screen text in their original language.
- Output ONLY the final optimized prompt. No explanations, no commentary.
{task_hint}
{extra}
=== OFFICIAL GUIDE ===
{guide}"""


def list_reference_files():
    root = os.path.join(folder_paths.base_path, "model_doc")
    if not os.path.isdir(root):
        return ["none"]
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith((".md", ".txt")):
                files.append(os.path.relpath(os.path.join(dirpath, name), root))
    return ["none"] + sorted(files)


def call_chat_completions(cfg, key, model, system, user_text, temperature):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 8192,
        "stream": False,
    }
    # reasoner-family models do not support temperature
    if "reasoner" not in model:
        payload["temperature"] = temperature

    response = requests.post(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek API error {response.status_code}: {response.text[:500]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def call_responses_api(cfg, key, model, system, user_text, temperature):
    payload = {
        "model": model,
        "instructions": system,
        "input": user_text,
        "tools": [{"type": "web_search"}],
        "temperature": temperature,
        "stream": False,
    }
    response = requests.post(
        cfg["base_url"].rstrip("/") + "/responses",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek Responses API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    texts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    content = "\n".join(t for t in texts if t).strip()
    if not content:
        raise RuntimeError(f"DeepSeek Responses API returned no text: {str(data)[:500]}")
    return content


class DeepSeekPromptOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["prompt_optimize", "chat"], {"default": "prompt_optimize"}),
                "guide": (["base", "ref", "base+ref"], {"default": "base"}),
                "task_type": (["auto", "T2VA", "I2VA", "FL2VA", "L2VA"], {"default": "auto"}),
                "model": (fetch_models(), {}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "web_search": ("BOOLEAN", {"default": False}),
                "reference_file": (list_reference_files(), {"default": "none"}),
            },
            "optional": {
                "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_prompt",)
    FUNCTION = "optimize"
    CATEGORY = "prompt"

    def optimize(self, prompt, mode, guide, task_type, model, temperature, web_search, reference_file, extra_instructions="", api_key=""):
        cfg = load_config()
        key = api_key.strip() or cfg["api_key"].strip()
        if not key:
            raise ValueError(f"DeepSeek API key is empty. Set it in {CONFIG_PATH} or in the node's api_key input.")

        if not prompt.strip():
            raise ValueError("prompt is empty.")

        ref_text = ""
        if reference_file != "none":
            root = os.path.abspath(os.path.join(folder_paths.base_path, "model_doc"))
            path = os.path.abspath(os.path.join(root, reference_file))
            if not path.startswith(root + os.sep) or not os.path.isfile(path):
                raise FileNotFoundError(f"Reference file not found under {root}: {reference_file}")
            with open(path, "r", encoding="utf-8") as f:
                ref_text = f.read()

        if mode == "prompt_optimize":
            guide_text = load_guides(guide)
            task_hint = "" if task_type == "auto" else f"- Task type: {task_type}. Follow the corresponding section of the guide."
            extra = "" if not extra_instructions.strip() else f"- Additional user instructions: {extra_instructions.strip()}"
            system = SYSTEM_TEMPLATE.format(task_hint=task_hint, extra=extra, guide=guide_text)
            if ref_text:
                system += f"\n\n=== ADDITIONAL REFERENCE ({reference_file}) ===\n{ref_text}"
        else:
            system = "You are a helpful assistant. Answer the user's question directly and concisely."
            if extra_instructions.strip():
                system += f"\nAdditional user instructions: {extra_instructions.strip()}"
            if ref_text:
                system += f"\n\nAnswer using the reference document below when it is relevant.\n=== REFERENCE DOCUMENT ({reference_file}) ===\n{ref_text}"

        if web_search:
            if model != "deepseek-v4-flash":
                raise ValueError(f"web_search is only available with the deepseek-v4-flash model, current model: {model}")
            content = call_responses_api(cfg, key, model, system, prompt.strip(), temperature)
        else:
            content = call_chat_completions(cfg, key, model, system, prompt.strip(), temperature)
        return (content,)


NODE_CLASS_MAPPINGS = {
    "DeepSeekPromptOptimizer": DeepSeekPromptOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekPromptOptimizer": "DeepSeek Prompt Optimizer (MiniMax H3)",
}
