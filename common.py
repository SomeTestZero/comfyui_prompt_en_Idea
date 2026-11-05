import json
import os
import time
from datetime import datetime

import requests

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(NODE_DIR, "config.json")
LOG_PATH = os.path.join(NODE_DIR, "h3_prompt_enhancer.log")
HISTORY_PATH = os.path.join(NODE_DIR, "prompt_history.jsonl")
HISTORY_MAX_BYTES = 8 * 1024 * 1024
HISTORY_KEEP_ON_TRIM = 1500
LOG_MAX_BYTES = 8 * 1024 * 1024
LOG_KEEP_LINES = 2000
LORA_PROFILES_PATH = os.path.join(NODE_DIR, "lora_contexts.json")
LORA_PRESETS_PATH = os.path.join(NODE_DIR, "lora_presets.json")


def log(msg):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        # Prompt dumps make this file grow forever otherwise; keep the recent tail.
        if os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                tail = f.readlines()[-LOG_KEEP_LINES:]
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                f.writelines(tail)
    except Exception:
        pass


def append_history(record):
    try:
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if os.path.getsize(HISTORY_PATH) > HISTORY_MAX_BYTES:
            entries = read_history()[-HISTORY_KEEP_ON_TRIM:]
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"history append failed: {e}")


def read_history():
    entries = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def history_label(e):
    task = str(e.get("task_type", "?")).split(" - ")[0]
    return f"{e.get('ts', '?')} | {task} | {e.get('input', '')[:40]}"


def history_entry_options(limit=100, kind="optimize"):
    labels = ["none"]
    for e in reversed([e for e in read_history() if e.get("kind") == kind][-limit:]):
        label = history_label(e)
        while label in labels:  # same-second generations can collide
            label += " "
        labels.append(label)
    return labels


def find_history_entry(label, kind="optimize"):
    for e in reversed(read_history()):
        if e.get("kind") == kind and history_label(e) == label.rstrip():
            return e
    return None


def clean_api_key(value):
    """Widget-value shifts in saved workflows can leave junk like 'False' in this field."""
    v = value.strip()
    return "" if v.lower() in ("", "false", "none", "null", "undefined") else v


def read_lora_profiles():
    if os.path.exists(LORA_PROFILES_PATH):
        try:
            with open(LORA_PROFILES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log(f"lora profiles read failed: {e}")
    return {}


def save_lora_profile(name, probe_type, context, trigger=""):
    profiles = read_lora_profiles()
    profiles[name] = {
        "context": context,
        "probe_type": probe_type,
        "trigger": trigger.strip(),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    with open(LORA_PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=1)


def lora_profile_options():
    return ["none"] + sorted(read_lora_profiles())


def read_lora_presets():
    if os.path.exists(LORA_PRESETS_PATH):
        try:
            with open(LORA_PRESETS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log(f"lora presets read failed: {e}")
    return {}


LORA_PRESET_NONE = "不注入"


def _no_preset(preset):
    # Older saves may still carry the English "none" sentinel from the first build.
    return preset in ("", "none", None, LORA_PRESET_NONE)


def lora_preset_options():
    """[不注入] + presets in file order, so the dropdown reads top-down by purpose."""
    return [LORA_PRESET_NONE] + list(read_lora_presets())


def build_preset_note(preset):
    """Instruction bullet for a LoRA preset; '' when none.

    Presets live in lora_presets.json, not in skills/: the official skill files
    stay byte-identical to MiniMax-H3 while LoRA specifics plug in here.
    """
    if _no_preset(preset):
        return ""
    p = read_lora_presets().get(preset)
    if p is None:
        raise ValueError(f"LoRA preset not found (list may be stale, reselect it): {preset}")
    note = f"\n- This generation runs with the LoRA preset \"{preset}\" active. {p.get('guidance', '').strip()}".rstrip()
    triggers = p.get("triggers") or []
    if triggers:
        note += ("\n- Activation phrases must survive verbatim in the final prompt, exact spelling and punctuation, "
                 "never translated or reworded: " + ", ".join(triggers) +
                 ". Put them in one short cue line at the end of detailed_description (e.g. \"Motion cues: ...\").")
    return note


def apply_preset_triggers(text, preset):
    """Guarantee the activation phrases are present; returns (text, added_phrases).

    The rewrite normally includes them already; this is the safety net for
    models that drop rare tokens. Cue line goes before the sound sections.
    """
    if _no_preset(preset):
        return text, []
    p = read_lora_presets().get(preset)
    if p is None:
        return text, []
    lowered = text.lower()
    missing = [t for t in (p.get("triggers") or []) if t.strip().lower() not in lowered]
    if not missing:
        return text, []
    line = "Motion cues: " + ", ".join(missing) + "."
    for marker in ("\noverall_soundscape:", "\nnon_diegetic_music:"):
        i = text.find(marker)
        if i != -1:
            return text[:i].rstrip() + "\n" + line + "\n" + text[i:], missing
    return text.rstrip() + "\n" + line, missing


def load_config():
    cfg = {"api_key": "", "base_url": "https://api.deepseek.com"}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


_model_list_cache = {"models": [], "fetched_at": 0.0}
MODEL_LIST_TTL = 3600

# Last generated output, reused when keep_previous_output is enabled
_last_generation = {"key": None, "output": None}


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


REQUEST_TIMEOUT = (30, 60)  # connect, per-chunk read (streaming)
REQUEST_RETRIES = 2


def post_with_retry(url, headers, payload):
    last_err = None
    for attempt in range(1, REQUEST_RETRIES + 2):
        try:
            return requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            log(f"request attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt <= REQUEST_RETRIES:
                time.sleep(5 * attempt)
    raise last_err


def make_progress_cb(unique_id):
    """Throttle-free caller side throttles; pushes partial text onto the node in the UI."""
    if not unique_id:
        return None
    try:
        from server import PromptServer
    except Exception:
        return None

    def cb(text):
        try:
            if PromptServer.instance:
                PromptServer.instance.send_progress_text(text, unique_id)
        except Exception:
            pass

    return cb


def call_chat_completions(cfg, key, model, system, user_text, temperature, on_text=None, thinking=False,
                          *, seed=None, max_tokens=8192, control_thinking=True, reasoning_effort=None):
    """user_text may be a string or an OpenAI-style content block list (multimodal)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if control_thinking:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if seed is not None:
        payload["seed"] = seed
    # thinking mode ignores temperature; v3-era reasoner models reject it
    if "reasoner" not in model:
        payload["temperature"] = temperature

    # OAuth providers pass a ready header set (and possibly a different path);
    # everything else keeps the plain bearer-token shape.
    url = cfg["base_url"].rstrip("/") + cfg.get("chat_path", "/chat/completions")
    headers = cfg.get("headers") or {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(1, REQUEST_RETRIES + 2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT, stream=True)
            if response.status_code != 200:
                raise RuntimeError(f"chat completions API error {response.status_code}: {response.text[:500]}")
            chunks, last_push = [], 0.0
            # iter_lines(decode_unicode=False) + manual utf-8 decode: str.splitlines
            # would also split on U+2028/U+2029, which JSON strings may contain
            # unescaped - thinking-heavy models (GLM) do emit them and would
            # break json.loads mid-line.
            for raw in response.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"bad SSE data line: {data[:200]!r}") from e
                piece = delta.get("content")  # reasoning_content is intentionally dropped
                if piece:
                    chunks.append(piece)
                    if on_text and time.time() - last_push > 0.25:
                        last_push = time.time()
                        on_text("".join(chunks))
            content = "".join(chunks).strip()
            if on_text:
                on_text(content)
            return content
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            log(f"chat attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt <= REQUEST_RETRIES:
                time.sleep(5 * attempt)
    raise last_err


def call_responses_api(cfg, key, model, system, user_text, temperature):
    payload = {
        "model": model,
        "instructions": system,
        "input": user_text,
        "tools": [{"type": "web_search"}],
        "temperature": temperature,
        "stream": False,
    }
    response = post_with_retry(
        cfg["base_url"].rstrip("/") + "/responses",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload,
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
