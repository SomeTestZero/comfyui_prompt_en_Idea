# -*- coding: utf-8 -*-
# ruff: noqa: T201
"""Offline tests for the cloud API nodes: provider/model resolution, payload
construction and message assembly. No network access - call_chat_completions
is stubbed at the module boundary."""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, r"E:\ComfyUI")
sys.path.insert(0, r"E:\ComfyUI\custom_nodes")

fake_server = types.ModuleType("server")
fake_server.PromptServer = None
sys.modules.setdefault("server", fake_server)

from h3_prompt_enhancer import nodes_cloud
from h3_prompt_enhancer.common import call_chat_completions
from h3_prompt_enhancer.nodes_cloud import (
    DEFAULT_MODEL,
    cloud_model_options,
    chat,
    image_part,
    model_info,
    parse_model_option,
    replay_entry_mode,
    require_vision,
    resolve_api_key,
    resolve_provider,
)


class FakeTensor:
    def __init__(self, tag):
        self.tag = tag


def with_config(cloud=None, top=None):
    """Temporarily point CONFIG_PATH at a temp config."""
    cfg = dict(top or {})
    if cloud is not None:
        cfg["cloud"] = cloud
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    old = nodes_cloud.load_config.__globals__["CONFIG_PATH"]
    nodes_cloud.load_config.__globals__["CONFIG_PATH"] = path
    return path, old


def test_parse_and_options():
    provider, model = parse_model_option("volcengine-plan/glm-5.3-flash")
    assert (provider, model) == ("volcengine-plan", "glm-5.3-flash")
    try:
        parse_model_option("no-slash")
        raise AssertionError("should raise")
    except ValueError:
        pass
    options = cloud_model_options()
    assert DEFAULT_MODEL in options and DEFAULT_MODEL == "volcengine-plan/glm-5.3-flash"
    assert "volcengine/doubao-seed-2-1-pro-260628" in options
    assert "volcengine-plan/glm-5.3-flash" in options
    assert "deepseek/deepseek-v4-flash-vision-exp" in options
    # config-added models and providers appear in the options
    path, old = with_config(cloud={"volcengine": {"models": ["my-ep-id"]},
                                   "siliconflow": {"base_url": "https://api.siliconflow.cn/v1",
                                                   "models": ["Qwen/Qwen3-VL-32B"]}})
    try:
        options = cloud_model_options()
        assert "volcengine/my-ep-id" in options
        assert "siliconflow/Qwen/Qwen3-VL-32B" in options
    finally:
        nodes_cloud.load_config.__globals__["CONFIG_PATH"] = old
        os.unlink(path)
    print("parse_and_options OK")


def test_resolve_provider_and_key():
    base, _ = resolve_provider("volcengine")
    assert base == "https://ark.cn-beijing.volces.com/api/v3"
    base, _ = resolve_provider("volcengine-plan")
    assert base == "https://ark.cn-beijing.volces.com/api/plan/v3"
    # custom provider override + missing base_url error
    path, old = with_config(cloud={"volcengine": {"base_url": "https://relay.example/v3/"},
                                   "bare": {"models": ["x"]}})
    try:
        base, _ = resolve_provider("volcengine")
        assert base == "https://relay.example/v3"  # trailing slash stripped
        try:
            resolve_provider("bare")
            raise AssertionError("should raise")
        except ValueError:
            pass
    finally:
        nodes_cloud.load_config.__globals__["CONFIG_PATH"] = old
        os.unlink(path)
    # key resolution: node > config cloud > legacy top-level (deepseek only) > env
    path, old = with_config(cloud={"volcengine": {"api_key": "cfg-key"}},
                            top={"api_key": "legacy-deepseek"})
    try:
        assert resolve_api_key("volcengine", " node-key ") == "node-key"
        assert resolve_api_key("volcengine", "") == "cfg-key"
        assert resolve_api_key("deepseek", "") == "legacy-deepseek"
    finally:
        nodes_cloud.load_config.__globals__["CONFIG_PATH"] = old
        os.unlink(path)
    # env fallback, then the clear empty-key error
    path, old = with_config(cloud={})
    try:
        os.environ["VOLCENGINE_API_KEY"] = "env-key"
        try:
            assert resolve_api_key("volcengine", "") == "env-key"
        finally:
            del os.environ["VOLCENGINE_API_KEY"]
        try:
            resolve_api_key("volcengine", "")
            raise AssertionError("should raise")
        except ValueError as e:
            assert "VOLCENGINE_API_KEY" in str(e)
    finally:
        nodes_cloud.load_config.__globals__["CONFIG_PATH"] = old
        os.unlink(path)
    print("resolve_provider_and_key OK")


def test_model_info_and_vision():
    info = model_info("volcengine", "doubao-seed-2-1-pro-260628")
    assert info["vision"] is True and info["thinking"] == "controlled" and info["effort"]["low"] == "low"
    info = model_info("volcengine-plan", "glm-5.3-flash")
    assert info["vision"] is True and info["thinking"] == "forced" and info["effort"]["medium"] == "high"
    # plan-endpoint entries verified against a real key: doubao vision on, deepseek vision off
    assert model_info("volcengine-plan", "doubao-seed-2-1-turbo-260628")["vision"] is True
    assert model_info("volcengine-plan", "kimi-k3")["vision"] is True
    assert model_info("volcengine-plan", "deepseek-v4-flash-vision-exp")["vision"] is False
    assert model_info("volcengine-plan", "minimax-m3")["vision"] is False
    info = model_info("deepseek", "deepseek-v4-flash-vision-exp")
    assert info["vision"] is True and "effort" not in info
    assert model_info("volcengine", "unknown-model") is None
    # unknown models: images allowed; known text models rejected with images
    require_vision("volcengine", "unknown-model", [("t", "reference")])
    try:
        require_vision("volcengine-plan", "glm-5.3", [("t", "reference")])
        raise AssertionError("should raise")
    except ValueError as e:
        assert "does not accept image input" in str(e)
    require_vision("volcengine-plan", "glm-5.3", [])  # no images, fine
    print("model_info_and_vision OK")


def test_image_part_and_replay():
    import torch
    img = torch.rand(1, 8, 8, 3)
    part = image_part(img)
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert replay_entry_mode({"task_type": "cloud/Ref2VA"}, "Auto") == "Ref2VA"
    assert replay_entry_mode({"task_type": "local/I2VA"}, "Auto") == "I2VA"
    assert replay_entry_mode({"task_type": "Auto - 自动判断"}, "L2VA") == "L2VA"
    assert replay_entry_mode({"task_type": ""}, "Auto") == "T2VA"
    print("image_part_and_replay OK")


def test_chat_payload():
    """Stub the HTTP layer and inspect the payload each node profile builds."""
    captured = {}

    class FakeResponse:
        status_code = 200
        def __init__(self, lines=None):
            self.lines = lines or [
                b'data: {"choices": [{"delta": {"content": "ok"}}]}',
                b"data: [DONE]",
            ]
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False  # str.splitlines would break on U+2028
            return iter(self.lines)

    def fake_post(url, headers=None, json=None, timeout=None, stream=True):
        captured["url"] = url
        captured["payload"] = json
        # SSE line containing a raw U+2028 (unescaped in JSON, emitted by
        # thinking-heavy models): must survive line splitting intact
        if json.get("seed") == 2028:
            tricky = b'data: {"choices": [{"delta": {"content": "a\xe2\x80\xa8b"}}]}'
            return FakeResponse([tricky, b"data: [DONE]"])
        return FakeResponse()

    import h3_prompt_enhancer.common as common
    old_post = common.requests.post
    common.requests.post = fake_post
    try:
        # doubao: thinking switch + effort + seed, multimodal content list
        content = chat("volcengine", "doubao-seed-2-1-pro-260628", "sys",
                       [{"type": "text", "text": "u"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
                       temperature=0.7, seed=42, thinking=True, effort_choice="low",
                       api_key="k", max_tokens=16384)
        assert content == "ok"
        p = captured["payload"]
        assert captured["url"] == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        assert p["model"] == "doubao-seed-2-1-pro-260628"
        assert p["max_tokens"] == 16384 and p["stream"] is True
        assert p["thinking"] == {"type": "enabled"}
        assert p["reasoning_effort"] == "low" and p["seed"] == 42 and p["temperature"] == 0.7
        assert p["messages"][1]["content"][1]["image_url"]["url"] == "data:x"
        # doubao with thinking disabled: effort dropped (Ark 400s on the combo,
        # verified against the plan endpoint: "low + disabled" InvalidParameter)
        chat("volcengine", "doubao-seed-2-1-pro-260628", "sys", "u",
             temperature=0.7, seed=None, thinking=False, effort_choice="low",
             api_key="k", max_tokens=1024)
        p = captured["payload"]
        assert p["thinking"] == {"type": "disabled"} and "reasoning_effort" not in p
        # glm: no thinking field, medium effort mapped to high
        chat("volcengine-plan", "glm-5.3-flash", "sys", "u", temperature=0.7, seed=1,
             thinking=False, effort_choice="medium", api_key="k", max_tokens=1024)
        p = captured["payload"]
        assert "thinking" not in p
        assert p["reasoning_effort"] == "high"
        assert p["messages"][1]["content"] == "u"
        # glm auto effort: not sent
        chat("volcengine-plan", "glm-5.3", "sys", "u", temperature=0.7, seed=None,
             thinking=True, effort_choice="auto", api_key="k", max_tokens=1024)
        p = captured["payload"]
        assert "reasoning_effort" not in p and "seed" not in p
        # deepseek: no effort, thinking switch sent, no seed when None
        chat("deepseek", "deepseek-v4-flash-vision-exp", "sys", "u", temperature=0.3,
             seed=None, thinking=True, effort_choice="high", api_key="k", max_tokens=8192)
        p = captured["payload"]
        assert p["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in p and "seed" not in p
        # legacy DeepSeek node path unchanged: text user, 8192, thinking switch
        content = call_chat_completions({"base_url": "https://api.deepseek.com"}, "k",
                                        "deepseek-v4-flash", "sys", "u", 0.7, thinking=True)
        p = captured["payload"]
        assert p["max_tokens"] == 8192 and p["thinking"] == {"type": "enabled"} and p["temperature"] == 0.7
        assert content == "ok"
        # U+2028 inside a streamed delta survives line splitting
        content = call_chat_completions({"base_url": "https://x"}, "k", "glm-5.3-flash",
                                        "sys", "u", 0.7, seed=2028)
        assert content == "a\u2028b", repr(content)
    finally:
        common.requests.post = old_post
    print("chat_payload OK")


if __name__ == "__main__":
    test_parse_and_options()
    test_resolve_provider_and_key()
    test_model_info_and_vision()
    test_image_part_and_replay()
    test_chat_payload()
    print("ALL CLOUD OFFLINE TESTS PASSED")
