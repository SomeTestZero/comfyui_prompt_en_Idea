# -*- coding: utf-8 -*-
# ruff: noqa: T201
"""Offline tests for the Qwen-Image-2.1 prompt enhancer: skill routing,
user-text/image-note assembly, and node execution with a stubbed chat()."""
import os
import sys
import tempfile
import types

sys.path.insert(0, r"E:\ComfyUI")
sys.path.insert(0, r"E:\ComfyUI\custom_nodes")

fake_server = types.ModuleType("server")
fake_server.PromptServer = None
sys.modules.setdefault("server", fake_server)

from h3_prompt_enhancer import common, nodes_qwen21
from h3_prompt_enhancer.nodes_qwen21 import build_user_text, collect_frames, derive_mode
from h3_prompt_enhancer.skills import load_skill, scan_skills


def test_skill_routing():
    assert "qwen-image21-prompt-writing" in scan_skills()
    body, refs = load_skill("qwen-image21-prompt-writing", "T2I")
    assert "Qwen-Image 2.1" in body
    assert [n for n, _ in refs] == [os.path.join("references", "t2i-en.txt")]
    assert "Image Prompt Rewriting Expert" in refs[0][1]
    _, refs = load_skill("qwen-image21-prompt-writing", "Edit")
    assert [n for n, _ in refs] == [os.path.join("references", "edit-en.txt")]
    assert "Edit Prompt Enhancer" in refs[0][1]
    _, refs = load_skill("qwen-image21-prompt-writing", "CharacterSheet")
    assert [n for n, _ in refs] == [os.path.join("references", "charactersheet-en.txt")]
    assert "CHARACTER REFERENCE SHEET" in refs[0][1]
    # no keyed guide for the mode -> every reference file
    _, refs = load_skill("qwen-image21-prompt-writing", "generic")
    assert {n for n, _ in refs} == {os.path.join("references", "t2i-en.txt"),
                                    os.path.join("references", "edit-en.txt"),
                                    os.path.join("references", "charactersheet-en.txt")}
    # h3 routing unchanged (base-en.txt / ref-en.txt win over keyed lookup)
    _, refs = load_skill("h3-prompt-writing", "T2VA")
    assert [n for n, _ in refs] == [os.path.join("references", "base-en.txt")]
    _, refs = load_skill("h3-prompt-writing", "Ref2VA")
    assert [n for n, _ in refs] == [os.path.join("references", "ref-en.txt")]
    _, refs = load_skill("krea2-prompt-writing", "generic")
    assert len(refs) == 2
    print("skill_routing OK")


def test_mode_and_frames():
    assert derive_mode("Auto", []) == "T2I"
    assert derive_mode("Auto", ["f"]) == "Edit"
    assert derive_mode("T2I", ["f"]) == "T2I"
    assert derive_mode("Edit", []) == "Edit"
    assert derive_mode("CharacterSheet", []) == "CharacterSheet"
    assert derive_mode("CharacterSheet", ["f"]) == "CharacterSheet"
    frames = collect_frames({"image_10": ["c"], "image_2": ["b"], "image_1": ["a", "a2"]})
    assert frames == ["a", "a2", "b", "c"]  # slot order, not dict order
    assert collect_frames(None) == []
    print("mode_and_frames OK")


def test_user_text():
    text, note = build_user_text("do a thing", [])
    assert text == "do a thing" and note == ""
    text, note = build_user_text("swap the shirt", ["f1", "f2"])
    assert text == "swap the shirt\n\n2 image(s) attached in order: <image1>, <image2>."
    assert "<image1>, <image2>" in note and "image_1 = <image1>" in note
    # empty prompt: intent reverse-inferred from the images
    text, note = build_user_text("", ["f1"])
    assert "Reverse-infer" in text and "<image1>" in text
    assert "\"the image\"" in note  # single image -> natural reference rule
    print("user_text OK")


def test_qwen21_node():
    import torch

    node = nodes_qwen21.QwenImage21PromptEnhancerCloud.PREPARE_CLASS_CLONE(None)
    calls = []

    def fake_chat(provider, model, system, user_content, **kw):
        calls.append({"provider": provider, "model": model, "system": system,
                      "user": user_content, "kw": kw})
        return f"prompt{len(calls)}"

    old_chat, old_path = nodes_qwen21.chat, common.HISTORY_PATH
    fd, hist = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    nodes_qwen21.chat = fake_chat
    common.HISTORY_PATH = hist
    try:
        # two images -> Edit mode, PE-I2I guide only, tags in user text
        out = node.execute(prompt="put the shirt on her", task_type="Auto",
                           skill="qwen-image21-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=1, reasoning_effort="low",
                           images={"image_1": [torch.rand(1, 8, 8, 3)],
                                   "image_2": [torch.rand(1, 8, 8, 3)]})
        assert out.result == ("prompt1",), out.result
        c = calls[0]
        assert (c["provider"], c["model"]) == ("volcengine-plan", "glm-5.3-flash")
        assert "Task mode: Edit." in c["system"]
        # guide bodies are routed: PE-I2I loaded, PE-T2I not
        assert "clarifying image editing instructions" in c["system"]
        assert "one long English paragraph" not in c["system"]
        assert "rewritten_prompt" in c["system"]  # JSON-envelope unwrap rule
        assert c["user"][0]["text"].endswith("<image1>, <image2>.")
        assert len(c["user"]) == 3
        assert c["user"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert c["kw"] == {"temperature": 0.7, "seed": 1, "thinking": False,
                           "effort_choice": "low", "api_key": "",
                           "max_tokens": nodes_qwen21.ENHANCE_MAX_TOKENS, "on_text": None}
        # history record, visible to the shared optimize dropdowns
        entries = common.read_history()
        assert len(entries) == 1
        assert entries[0]["task_type"] == "cloud/qwen21/Edit"
        assert entries[0]["output"] == "prompt1"
        # no images -> T2I mode, PE-T2I guide only
        out = node.execute(prompt="a teapot", task_type="Auto",
                           skill="qwen-image21-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="enabled",
                           temperature=0.7, seed=2)
        assert out.result == ("prompt2",)
        c = calls[1]
        assert "Task mode: T2I." in c["system"]
        assert "one long English paragraph" in c["system"]
        assert "clarifying image editing instructions" not in c["system"]
        assert len(c["user"]) == 1
        assert c["kw"]["thinking"] is True
        # replay: no API call, stored output comes back
        labels = nodes_qwen21.history_entry_options()
        assert len(labels) == 3 and labels[2].endswith("cloud/qwen21/Edit | put the shirt on her")
        out = node.execute(prompt="", task_type="Auto", skill="qwen-image21-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=0, history_entry=labels[2])
        assert out.result == ("prompt1",) and len(calls) == 2
        # text-only model never receives the image
        try:
            node.execute(prompt="x", task_type="Auto", skill="qwen-image21-prompt-writing",
                         model="volcengine-plan/glm-5.3", thinking="disabled",
                         temperature=0.7, seed=0,
                         images={"image_1": [torch.rand(1, 8, 8, 3)]})
            raise AssertionError("should raise")
        except ValueError as err:
            assert "does not accept image input" in str(err)
        assert len(calls) == 2
        # empty prompt and no images -> clear error
        try:
            node.execute(prompt="  ", task_type="Auto", skill="qwen-image21-prompt-writing",
                         model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                         temperature=0.7, seed=0)
            raise AssertionError("should raise")
        except ValueError as err:
            assert "no reference images" in str(err)
        assert len(calls) == 2
        # character-sheet mode: sheet guide routes in, with or without images
        out = node.execute(prompt="dante reference sheet", task_type="CharacterSheet",
                           skill="qwen-image21-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=3)
        assert out.result == ("prompt3",)
        c = calls[2]
        assert "Task mode: CharacterSheet." in c["system"]
        assert "one-inventory rule" in c["system"]
        assert "clarifying image editing instructions" not in c["system"]
        assert "one long English paragraph" not in c["system"]
        assert "Background: transparent." not in c["system"]
        assert len(calls) == 3
        # background=Transparent -> official RGBA wrap, no backdrop
        out = node.execute(prompt="a white-haired swordswoman", task_type="CharacterSheet",
                           skill="qwen-image21-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=4, background="Transparent")
        assert out.result == ("prompt4",)
        c = calls[3]
        assert "This is an RGBA image with transparency." in c["system"]
        assert "Background: transparent." in c["system"]
        assert len(calls) == 4
    finally:
        nodes_qwen21.chat, common.HISTORY_PATH = old_chat, old_path
        os.unlink(hist)
    print("qwen21_node OK")


if __name__ == "__main__":
    test_skill_routing()
    test_mode_and_frames()
    test_user_text()
    test_qwen21_node()
    print("ALL QWEN21 OFFLINE TESTS PASSED")
