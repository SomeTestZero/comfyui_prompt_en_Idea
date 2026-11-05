# -*- coding: utf-8 -*-
# ruff: noqa: T201
"""Offline tests for the H3 character-replacement prompt node: skill routing,
sampling, user-text assembly, lint, and node execution with a stubbed chat().
No network access - chat() is stubbed at the module boundary."""
import os
import sys
import tempfile
import types

sys.path.insert(0, r"E:\ComfyUI")
sys.path.insert(0, r"E:\ComfyUI\custom_nodes")

fake_server = types.ModuleType("server")
fake_server.PromptServer = None
sys.modules.setdefault("server", fake_server)

from h3_prompt_enhancer import common, nodes_swap
from h3_prompt_enhancer.nodes_swap import (
    build_swap_user_text,
    build_target_check,
    lint_swap_prompt,
    sample_indices,
)
from h3_prompt_enhancer.skills import load_skill, scan_skills


def test_skill_routing():
    assert "h3-prompt-writing" in scan_skills()
    body, refs = load_skill("h3-prompt-writing", "CharacterSwap")
    assert "Character Replacement" in refs[0][1]
    assert "Anti-copy clause" in refs[0][1]
    # the swap guide must not bleed into the Ref2VA route
    _, refs = load_skill("h3-prompt-writing", "Ref2VA")
    assert all("Character Replacement" not in t for _, t in refs)
    print("skill_routing OK")


def test_sample_indices():
    assert sample_indices(3) == [0, 1, 2]
    idx = sample_indices(124)
    assert idx[0] == 0 and idx[-1] == 123 and len(idx) == 6 and idx == sorted(set(idx))
    print("sample_indices OK")


def test_user_text():
    text = build_swap_user_text("replace the man in purple", 2,
                                ["t=0.00s", "t=5.00s"], 5.0)
    assert text.startswith("replace the man in purple")
    assert "<Picture 1>, <Picture 2>" in text
    assert "t=0.00s" in text and "t=5.00s" in text
    assert "Target runtime: 5 seconds." in text
    text = build_swap_user_text("swap her", 1, ["frame 0"], 0.0)
    assert "do not stamp shot times" in text
    for kw, err in (("", "instruction is empty"),
                    ("x", "no character reference images"),
                    ):
        try:
            build_swap_user_text(kw, 0 if kw else 1, ["t=0.00s"], 5.0)
            raise AssertionError("should raise")
        except ValueError as e:
            assert err in str(e), (err, str(e))
    try:
        build_swap_user_text("x", 1, [], 5.0)
        raise AssertionError("should raise")
    except ValueError as e:
        assert "no source video frames" in str(e)
    print("user_text OK")


def test_lint():
    good = ("subject_definitions:\nx\nsummary:\ny\nretention_analysis:\nz\n"
            "detailed_description:\nw\noverall_soundscape:\na\nnon_diegetic_music:\nN/A\n"
            "<Subject 1> <Subject 2> <Video 1> <Picture 1>")
    assert lint_swap_prompt(good, 1) == []
    warns = lint_swap_prompt("summary:\nonly this", 1)
    assert any("missing section" in w for w in warns)
    assert any("<Subject 1>" in w for w in warns)
    warns = lint_swap_prompt(good.replace("<Picture 1>", "<Picture 3>"), 1)
    assert any("only 1 reference image" in w for w in warns)
    # target_check: <Subject 2> line + lint verdict
    assert good.count("<Subject 2>") >= 1
    check = build_target_check("subject_definitions:\n<Subject 2> is the man in purple.\nsummary:\n", 1)
    assert check.startswith("<Subject 2> is the man in purple.")
    assert "lint:" in check
    check = build_target_check("subject_definitions:\n<Subject 1> is the hero.\n<Subject 2> is P1: the left one.\nsummary:\n", 1)
    assert "<Subject 1> is the hero." in check and "<Subject 2> is P1" in check  # 角色表全量输出
    check = build_target_check("summary:\nnothing useful", 1)
    assert "anchoring unknown" in check
    print("lint OK")


def test_swap_node():
    import torch

    node = nodes_swap.H3SwapPromptCloud.PREPARE_CLASS_CLONE(None)
    calls = []

    def fake_chat(provider, model, system, user_content, **kw):
        calls.append({"provider": provider, "model": model, "system": system,
                      "user": user_content, "kw": kw})
        return ("subject_definitions:\n<Subject 1> is the character in <Picture 1>.\n"
                "<Subject 2> is the woman in the red jacket.\n<Video 1> is the source video.\nsummary:\nS\nretention_analysis:\nR\n"
                "detailed_description:\nD\noverall_soundscape:\nA\nnon_diegetic_music:\nN/A")

    old_chat, old_path = nodes_swap.chat, common.HISTORY_PATH
    fd, hist = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    nodes_swap.chat = fake_chat
    common.HISTORY_PATH = hist
    try:
        out = node.execute(
            instruction="replace the woman in the red jacket with the character in the pictures",
            model="volcengine-plan/glm-5.3-flash", thinking="disabled",
            temperature=0.7, seed=1,
            ref_images={"ref_image_1": [torch.rand(1, 8, 8, 3)]},
            source_frames=torch.rand(124, 8, 8, 3),
            duration=5.0)
        assert out.result[0].startswith("subject_definitions:")
        assert "<Subject 2> is the woman in the red jacket." in out.result[1] and "lint: OK" in out.result[1]
        c = calls[0]
        assert (c["provider"], c["model"]) == ("volcengine-plan", "glm-5.3-flash")
        assert "Task mode: CharacterSwap" in c["system"]
        assert "Anti-copy clause" in c["system"]  # swap guide routed
        assert "Full-Reference Mode Rewrite" not in c["system"]  # generic ref guide kept out
        user = c["user"]
        assert len(user) == 1 + 1 + 6  # text + 1 picture + 6 sampled frames
        assert "<Picture 1>" in user[0]["text"]
        assert "t=0.00s" in user[0]["text"] and "t=5.00s" in user[0]["text"]
        entries = common.read_history()
        assert entries[0]["task_type"] == "cloud/characterswap"
        assert entries[0]["kind"] == "swap"
        # empty instruction / missing sockets -> clear errors
        try:
            node.execute(instruction="  ", model="volcengine-plan/glm-5.3-flash",
                         thinking="disabled", temperature=0.7, seed=0,
                         ref_images={"ref_image_1": [torch.rand(1, 8, 8, 3)]},
                         source_frames=torch.rand(5, 8, 8, 3))
            raise AssertionError("should raise")
        except ValueError as e:
            assert "instruction is empty" in str(e)
        try:
            node.execute(instruction="swap", model="volcengine-plan/glm-5.3-flash",
                         thinking="disabled", temperature=0.7, seed=0,
                         source_frames=torch.rand(5, 8, 8, 3))
            raise AssertionError("should raise")
        except ValueError as e:
            assert "no character reference images" in str(e)
        assert len(calls) == 1
    finally:
        nodes_swap.chat, common.HISTORY_PATH = old_chat, old_path
        os.unlink(hist)
    print("swap_node OK")


if __name__ == "__main__":
    test_skill_routing()
    test_sample_indices()
    test_user_text()
    test_lint()
    test_swap_node()
    print("ALL SWAP OFFLINE TESTS PASSED")
