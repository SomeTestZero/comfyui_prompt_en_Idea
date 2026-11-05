"""WorkBuddy (CodeBuddy) OAuth cloud backend for the prompt-enhancer nodes.

WorkBuddy's OpenAI-compatible endpoint authenticates with a short-lived OAuth
bearer token, not a static API key, and the token is only valid for ~28 days.
Rather than asking the user to paste a token every month, this module reads the
same credential file pi/CodeBuddy keep (`auth.json`), sends the token, and on a
401 refreshes it through the plugin refresh endpoint - writing the rotated pair
back so the CLI and ComfyUI stay logged in together.

Everything here mirrors extensions/workbuddy.ts in the user's pi config:
same base URL, same attribution headers (usage is billed to the WorkBuddy
client only when those are present), same refresh handshake.
"""

import base64
import json
import os
import time

import requests

WB_BASE = "https://copilot.tencent.com/v2"
WB_ORIGIN = "https://www.codebuddy.cn"
WB_VERSION = "5.8.0"
WB_CHAT_UA = f"WorkBuddy/{WB_VERSION} WorkBuddy/{WB_VERSION} CLI/2.137.1"
WB_AUTH_UA = "CLI/2.63.2 CodeBuddy/2.63.2"
REFRESH_MARGIN = 5 * 60  # refresh this long before the token actually dies

DEFAULT_CREDENTIAL_PATH = os.path.join("~", ".pi", "agent", "auth.json")


def credential_path(config_path=None):
    return os.path.expanduser(config_path or DEFAULT_CREDENTIAL_PATH)


def _load_credential(config_path=None):
    """The `workbuddy` block of auth.json, or {} when the user never logged in."""
    path = credential_path(config_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("workbuddy", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_credential(credential, config_path=None):
    """Write the rotated token pair back, keeping every other provider intact."""
    path = credential_path(config_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return  # never overwrite a file we could not parse
    data["workbuddy"] = credential
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _jwt_expires(token):
    """Seconds-since-epoch from the JWT `exp` claim, or None if unreadable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except (IndexError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _refresh(credential, config_path=None):
    """Trade the refresh token for a new pair; returns the updated credential dict."""
    refresh_token = credential.get("refresh", "")
    if not refresh_token:
        raise RuntimeError(
            "WorkBuddy credentials have no refresh token - run `/login workbuddy` in pi once."
        )
    headers = {
        "accept": "application/json, text/plain, */*",
        "x-requested-with": "XMLHttpRequest",
        "origin": WB_ORIGIN,
        "referer": f"{WB_ORIGIN}/",
        "user-agent": WB_AUTH_UA,
        "accept-language": "zh-CN",
        "x-codebuddy-request": "1",
        "x-refresh-token": refresh_token,
        "x-auth-refresh-source": "plugin",
    }
    res = requests.post(
        f"{WB_BASE}/plugin/auth/token/refresh", headers=headers, json={}, timeout=(30, 60)
    )
    try:
        parsed = res.json()
    except ValueError:
        raise RuntimeError(f"WorkBuddy refresh returned non-JSON (HTTP {res.status_code})") from None
    if not res.ok:
        raise RuntimeError(
            f"WorkBuddy refresh rejected (HTTP {res.status_code}): {str(parsed.get('msg', ''))[:160]}"
        )
    if isinstance(parsed.get("code"), int) and parsed["code"] != 0:
        raise RuntimeError(
            f"WorkBuddy refresh failed (code {parsed['code']}): {str(parsed.get('msg', ''))[:160]}"
        )
    data = parsed.get("data") or {}
    access = data.get("accessToken")
    if not access:
        raise RuntimeError("WorkBuddy refresh returned no accessToken")
    expires_in = data.get("expiresIn") if isinstance(data.get("expiresIn"), (int, float)) else 28 * 86400
    new = dict(credential)
    new["access"] = access
    new["refresh"] = data.get("refreshToken") or refresh_token  # the pair rotates together
    new["expires"] = int((time.time() + expires_in - REFRESH_MARGIN) * 1000)
    if data.get("domain"):
        new["domain"] = data["domain"]
    _save_credential(new, config_path)
    return new


def access_token(config_path=None, force_refresh=False):
    """A currently-valid bearer token, refreshing first when it is close to expiry."""
    credential = _load_credential(config_path)
    token = credential.get("access", "")
    if not token:
        raise RuntimeError(
            "WorkBuddy is not logged in - run `/login workbuddy` in pi once "
            f"(expected credentials in {credential_path(config_path)})."
        )
    # `expires` is written by pi as ms-since-epoch; fall back to the JWT claim.
    expires_ms = credential.get("expires")
    expires = (float(expires_ms) / 1000.0) if isinstance(expires_ms, (int, float)) else _jwt_expires(token)
    if force_refresh or (expires is not None and expires - REFRESH_MARGIN < time.time()):
        token = _refresh(credential, config_path)["access"]
    return token


def chat_headers(token):
    """Attribution headers: without these the request is not billed to WorkBuddy."""
    return {
        "content-type": "application/json",
        "accept": "application/json, text/plain, */*",
        "authorization": f"Bearer {token}",
        "user-agent": WB_CHAT_UA,
        "origin": WB_ORIGIN,
        "referer": f"{WB_ORIGIN}/",
        "x-requested-with": "XMLHttpRequest",
        "accept-language": "zh-CN",
        "x-agent-purpose": "conversation",
        "x-agent-intent": "craft",
        "x-ide-name": "WorkBuddy",
        "x-ide-type": "WorkBuddy",
        "x-ide-version": WB_VERSION,
        "x-product": "SaaS",
        "x-private-data": "true",
        "x-codebuddy-request": "1",
    }


def chat_url():
    return f"{WB_BASE}/chat/completions"
