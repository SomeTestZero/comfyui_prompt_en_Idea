# -*- coding: utf-8 -*-
# ruff: noqa: T201
"""Offline tests for the MOSS-Image prompt enhancer: skill routing, layout draft
compilation/validation, caption cleanup, and node execution with a stubbed chat()."""
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

from h3_prompt_enhancer import common, nodes_moss
from h3_prompt_enhancer.moss_prompt import (
    build_system,
    build_user_text,
    compile_layout,
    normalize_caption,
    postprocess,
)
from h3_prompt_enhancer.skills import load_skill, scan_skills

DRAFT = {
    "width": 1024,
    "height": 1024,
    "bbox_coord_space": "abs",
    "bbox_format": "xyxy",
    "summary": {"caption": "A poster with a cup and a title."},
    "items": [
        {"label": "object", "caption": "A ceramic cup.", "bbox": [100, 200, 500, 600]},
        {"label": "text", "caption": "Big title.", "text": "MOSS", "bbox": [300, 100, 700, 180]},
    ],
}


def test_skill_routing():
    assert "moss-image-prompt-writing" in scan_skills()
    body, refs = load_skill("moss-image-prompt-writing", "T2I")
    assert "MOSS-Image" in body
    assert [n for n, _ in refs] == [os.path.join("references", "t2i-en.txt")]
    assert "finished images" in refs[0][1]
    body, refs = load_skill("moss-image-prompt-writing", "Layout")
    assert [n for n, _ in refs] == [os.path.join("references", "layout-en.txt")]
    assert "abs_centralized" in refs[0][1]
    print("skill_routing OK")


def test_compile_layout():
    compiled = json.loads(compile_layout(json.dumps(DRAFT), 1024, 1024))
    assert compiled["bbox_coord_space"] == "abs_centralized"
    assert compiled["bbox_format"] == "yxyx"
    assert compiled["summary"]["caption"] == "A poster with a cup and a title."
    assert compiled["items"][0]["bbox"] == [-312, -412, 88, -12]
    assert compiled["items"][1]["text"] == "MOSS"
    # fenced JSON is tolerated, fractional edges floor/ceil before centring
    fenced = "```json\n" + json.dumps(DRAFT).replace("[100, 200, 500, 600]", "[100.4, 200.6, 500.2, 600.9]") + "\n```"
    compiled = json.loads(compile_layout(fenced, 1024, 1024))
    assert compiled["items"][0]["bbox"] == [-312, -412, 89, -11]
    print("compile_layout OK")


def test_compile_layout_errors():
    cases = [
        (DRAFT, 1536, 1024, "does not match"),
        ({**DRAFT, "items": [{"label": "object", "caption": "cup", "bbox": [0.1, 0.2, 0.5, 0.6]}]}, 1024, 1024, "normalised"),
        ({**DRAFT, "items": [{"label": "object", "caption": "cup", "bbox": [900, 200, 1100, 600]}]}, 1024, 1024, "outside"),
        ({**DRAFT, "items": [{"label": "text", "caption": "title", "bbox": [300, 100, 700, 180]}]}, 1024, 1024, "exact visible text"),
        ({**DRAFT, "items": []}, 1024, 1024, "items"),
        ({**DRAFT, "summary": {}}, 1024, 1024, "summary.caption"),
        ({**DRAFT, "items": [{"label": "logo", "caption": "x", "bbox": [1, 1, 10, 10]}]}, 1024, 1024, "label"),
    ]
    for draft, width, height, needle in cases:
        try:
            compile_layout(json.dumps(draft), width, height)
            raise AssertionError(f"should raise: {needle}")
        except ValueError as err:
            assert needle in str(err), (needle, str(err))
    # canvas must be a positive multiple of 16
    try:
        compile_layout(json.dumps(DRAFT), 1000, 1024)
        raise AssertionError("should raise")
    except ValueError as err:
        assert "multiple of 16" in str(err)
    print("compile_layout_errors OK")


def test_caption_cleanup():
    assert normalize_caption('  "A quiet street."  ') == "A quiet street."
    assert normalize_caption("```\nA quiet street.\n```") == "A quiet street."
    assert postprocess(" plain caption ", "T2I", 1024, 1024) == "plain caption"
    print("caption_cleanup OK")


def test_system_prompt():
    body, refs = load_skill("moss-image-prompt-writing", "Layout")
    system = build_system(body, refs, "Layout", 1536, 1024)
    assert "Task mode: Layout" in system
    assert "1536x1024" in system
    assert "abs_centralized" in system
    body, refs = load_skill("moss-image-prompt-writing", "T2I")
    system = build_system(body, refs, "T2I", 1024, 1024)
    assert "Task mode: T2I" in system
    assert "centralised" not in system
    assert build_user_text(" a poster ", "Layout", 1024, 768).endswith("Canvas: 1024x768 pixels.")
    try:
        build_user_text("  ", "T2I", 1024, 1024)
        raise AssertionError("should raise")
    except ValueError as err:
        assert "empty" in str(err)
    print("system_prompt OK")


def test_moss_node():
    node = nodes_moss.MOSSPromptEnhancerCloud.PREPARE_CLASS_CLONE(None)
    calls = []

    def fake_chat(provider, model, system, user_content, **kw):
        calls.append({"provider": provider, "model": model, "system": system,
                      "user": user_content, "kw": kw})
        if "Task mode: Layout" in system:
            return json.dumps(DRAFT)
        return ' "A quiet street at dawn." '

    old_chat, old_path = nodes_moss.chat, common.HISTORY_PATH
    fd, hist = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    nodes_moss.chat = fake_chat
    common.HISTORY_PATH = hist
    try:
        out = node.execute(prompt="一张海报，杯子+大标题 MOSS", task_type="版式布局", width=1024,
                           height=1024, skill="moss-image-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=1)
        compiled = json.loads(out.result[0])
        assert compiled["items"][0]["bbox"] == [-312, -412, 88, -12]
        assert compiled["items"][1]["text"] == "MOSS"
        assert "Task mode: Layout" in calls[0]["system"]
        assert calls[0]["user"][0]["text"].endswith("Canvas: 1024x1024 pixels.")

        out = node.execute(prompt="清晨安静的街道", task_type="文生图", width=1024, height=1024,
                           skill="moss-image-prompt-writing",
                           model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                           temperature=0.7, seed=2)
        assert out.result == ("A quiet street at dawn.",)
        assert "Task mode: T2I" in calls[1]["system"]
        assert "finished images" in calls[1]["system"]

        entries = common.read_history()
        assert entries[0]["task_type"] == "cloud/moss/layout"
        assert entries[1]["task_type"] == "cloud/moss/t2i"
        assert entries[0]["canvas"] == "1024x1024"

        try:
            node.execute(prompt="  ", task_type="文生图", width=1024, height=1024,
                         skill="moss-image-prompt-writing",
                         model="volcengine-plan/glm-5.3-flash", thinking="disabled",
                         temperature=0.7, seed=0)
            raise AssertionError("should raise")
        except ValueError as err:
            assert "empty" in str(err)
        assert len(calls) == 2
    finally:
        nodes_moss.chat, common.HISTORY_PATH = old_chat, old_path
        os.unlink(hist)
    print("moss_node OK")


if __name__ == "__main__":
    test_skill_routing()
    test_compile_layout()
    test_compile_layout_errors()
    test_caption_cleanup()
    test_system_prompt()
    test_moss_node()
    print("ALL PASS")
