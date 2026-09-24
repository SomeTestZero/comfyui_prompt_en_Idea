# -*- coding: utf-8 -*-
# ruff: noqa: T201
"""Offline tests for the Krea 2 prompt enhancer: skill routing, user-text
assembly, and node execution with a stubbed chat()."""
import os
import sys
import tempfile
import types

sys.path.insert(0, r"E:\ComfyUI")
sys.path.insert(0, r"E:\ComfyUI\custom_nodes")

fake_server = types.ModuleType("server")
fake_server.PromptServer = None
sys.modules.setdefault("server", fake_server)

from h3_prompt_enhancer import common, nodes_krea2
from h3_prompt_enhancer.nodes_krea2 import build_user_text
from h3_prompt_enhancer.skills import load_skill, scan_skills


def test_skill_routing():
    assert "krea2-prompt-writing" in scan_skills()
    body, refs = load_skill("krea2-prompt-writing", "generic")
    assert "Krea 2" in body
    assert {n for n, _ in refs} == {os.path.join("references", "official-prompting.md"),
                                    os.path.join("references", "expansion.txt")}
    print("skill_routing OK")


def test_user_text():
    text, note = build_user_text("a teapot", [])
    assert text == "a teapot" and note == ""
    text, note = build_user_text("paint her at dusk", ["f1", "f2"])
    assert text.endswith("<image1>, <image2>.")
    assert "Describe only what is actually visible" in note
    text, note = build_user_text("", ["f1"])
    assert "Reverse-infer" in text and "<image1>" in text
    print("user_text OK")


def test_krea2_node():
    import torch

    node = nodes_krea2.Krea2PromptEnhancerCloud.PREPARE_CLASS_CLONE(None)
    calls = []

    def fake_chat(provider, model, system, user_content, **kw):
        calls.append({"provider": provider, "model": model, "system": system,
                      "user": user_content, "kw": kw})
        return f"prompt{len(calls)}"

    old_chat, old_path = nodes_krea2.chat, common.HISTORY_PATH
    fd, hist = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    nodes_krea2.chat = fake_chat
    common.HISTORY_PATH = hist
    try:
        out = node.execute(prompt="a swordswoman", skill="krea2-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=1, style_preset="仙侠水墨",
                           images={"image_1": [torch.rand(1, 8, 8, 3)]})
        assert out.result == ("prompt1",)
        c = calls[0]
        assert (c["provider"], c["model"]) == ("volcengine-plan", "glm-5.3-flash")
        assert "exactly one English paragraph" in c["system"]
        assert "expansion" in c["system"]  # both official refs routed
        assert "wuxia xianxia fantasy" in c["system"]  # pinned preset block
        assert "Insert it VERBATIM" in c["system"]
        assert "text alone" in c["system"]  # reference images ground only
        assert c["user"][0]["text"].endswith("<image1>.")
        assert len(c["user"]) == 2
        entries = common.read_history()
        assert entries[0]["task_type"] == "cloud/krea2/t2i"
        # no images + custom style box pinned the same way
        out = node.execute(prompt="a teapot", skill="krea2-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=2, style_profile="rice-paper texture")
        assert out.result == ("prompt2",)
        assert "rice-paper texture" in calls[1]["system"]
        assert len(calls[1]["user"]) == 1
        # empty prompt and no images -> clear error
        try:
            node.execute(prompt="  ", skill="krea2-prompt-writing",
                         model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                         temperature=0.7, seed=0)
            raise AssertionError("should raise")
        except ValueError as err:
            assert "no reference images" in str(err)
        assert len(calls) == 2
    finally:
        nodes_krea2.chat, common.HISTORY_PATH = old_chat, old_path
        os.unlink(hist)
    print("krea2_node OK")


if __name__ == "__main__":
    test_skill_routing()
    test_user_text()
    test_krea2_node()
    print("ALL KREA2 OFFLINE TESTS PASSED")
