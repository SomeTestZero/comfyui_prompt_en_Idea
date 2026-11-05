"""Shared MOSS-Image prompt-enhancer helpers: skill routing, the output contract,
caption cleanup, and the Layout draft -> compiled prompt conversion.

The official skill writes a draft with width/height, `bbox_coord_space="abs"` and
top-left pixel `xyxy` boxes, then compiles it with a bundled Python tool. The node
compiles the same draft here, matching comfy_extras/nodes_moss_image.py:
floor(left/top) and ceil(right/bottom), then subtract the canvas centre, giving
`abs_centralized` / `yxyx` boxes in a compact one-line JSON prompt.
"""
import json
import math
import re

TASK_TYPES = {"文生图": "T2I", "版式布局": "Layout"}
LABELS = {"object", "obj", "character", "text"}

SYSTEM_TEMPLATE = """You are a professional prompt engineer for the MOSS-Image text-to-image model (5B single-stream DiT, FLUX.2 small VAE, Qwen3.5-9B text encoder). Rewrite the user's request into the final MOSS-Image prompt, strictly following the installed skill and the writing guide below.

Rules:
- Task mode: {mode} — {deliverable}.
- The task mode fixes the output format: never switch to the other mode's deliverable (no layout JSON in caption mode, no prose in layout mode).
- Keep every subject, count, action, relationship, medium, style, mood, colour, visible wording and exclusion the user specified. Never contradict or drop them. Fill the creative gaps with concrete visible choices; invent no real-world facts the user did not give.
- Write the description in English. Any visible text stays in its original language, quoted exactly, keeping spelling, capitalisation, punctuation and line breaks; never translate it.
- Describe only what can be seen: no sounds, smells, backstory, intentions, process notes, camera settings or coordinates.
- No quality tags ("masterpiece", "best quality", "8k", "highly detailed") and no negative prompt.
- Never mention attached images, the resolution, the aspect ratio or pixel counts.{layout_note}

=== SKILL ===
{skill_body}

=== WRITING GUIDE: {ref_names} ===
{ref_text}"""

DELIVERABLES = {
    "T2I": "one cohesive caption in plain prose, 80-200 words, no headings or JSON",
    "Layout": "exactly one JSON object (the layout draft), no prose around it",
}

LAYOUT_NOTE = (
    "\n- Layout mode: the canvas is {width}x{height} pixels. Use exactly that canvas in the draft and give every box as "
    "[x_min, y_min, x_max, y_max] pixels from the top-left corner. Never emit the model's centralised/negative-coordinate "
    "form; this node converts the boxes and validates them."
)


def build_system(skill_body, refs, mode, width, height):
    return SYSTEM_TEMPLATE.format(
        mode=mode,
        deliverable=DELIVERABLES.get(mode, DELIVERABLES["T2I"]),
        layout_note=LAYOUT_NOTE.format(width=width, height=height) if mode == "Layout" else "",
        skill_body=skill_body,
        ref_names=" + ".join(n for n, _ in refs) or "none",
        ref_text="\n\n".join(t for _, t in refs),
    )


def build_user_text(prompt, mode, width, height):
    text = prompt.strip()
    if not text:
        raise ValueError("prompt is empty: describe the image you want first.")
    if mode == "Layout":
        text += f"\n\nCanvas: {width}x{height} pixels."
    return text


def validate_canvas(width, height):
    for name, value in (("width", width), ("height", height)):
        if int(value) <= 0 or int(value) % 16 != 0:
            raise ValueError(f"{name} must be a positive multiple of 16, got {value}.")
    if width > 16384 or height > 16384:
        raise ValueError(f"canvas {width}x{height} exceeds the 16384 pixel limit.")


def _unwrap_json(text):
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the layout draft")
    return json.loads(body[start:end + 1])


def _bbox(item, index, width, height):
    box = item.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"item {index}: bbox must be four numbers [x_min, y_min, x_max, y_max]")
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        raise ValueError(f"item {index}: bbox values must be numbers, got {box!r}")
    if all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        raise ValueError(f"item {index}: bbox {box!r} looks normalised; use pixel coordinates from the top-left corner")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"item {index}: bbox {box!r} is outside the {width}x{height} canvas or empty")
    # top-left pixels -> abs_centralized yxyx, floored/ceiled before centring
    return [
        math.floor(y0) - height // 2,
        math.floor(x0) - width // 2,
        math.ceil(y1) - height // 2,
        math.ceil(x1) - width // 2,
    ]


def compile_layout(text, width, height):
    """Validate a layout draft and return the compiled one-line prompt."""
    validate_canvas(width, height)
    draft = _unwrap_json(text)
    if not isinstance(draft, dict):
        raise ValueError("layout draft must be a JSON object")

    draft_width = draft.get("width")
    draft_height = draft.get("height")
    if draft_width is not None and int(draft_width) != width:
        raise ValueError(f"layout draft width {draft_width} does not match the canvas width {width}")
    if draft_height is not None and int(draft_height) != height:
        raise ValueError(f"layout draft height {draft_height} does not match the canvas height {height}")

    summary = draft.get("summary")
    caption = summary.get("caption") if isinstance(summary, dict) else None
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("layout draft needs a non-empty summary.caption")

    items = draft.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("layout draft needs a non-empty items array")

    compiled = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"item {index}: must be an object")
        label = str(item.get("label", "object")).lower()
        if label not in LABELS:
            raise ValueError(f"item {index}: label must be object, character or text, got {label!r}")
        if label == "obj":
            label = "object"
        item_caption = item.get("caption")
        if not isinstance(item_caption, str) or not item_caption.strip():
            raise ValueError(f"item {index}: caption is required")
        entry = {"label": label, "caption": item_caption.strip()}
        if label == "text":
            visible = item.get("text")
            if not isinstance(visible, str) or not visible.strip():
                raise ValueError(f"item {index}: text items need the exact visible text in `text`")
            entry["text"] = visible
        palette = item.get("palette")
        if isinstance(palette, list) and palette:
            entry["palette"] = [str(colour) for colour in palette]
        entry["bbox"] = _bbox(item, index, width, height)
        compiled.append(entry)

    layout = {
        "summary": {"caption": caption.strip()},
        "items": compiled,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def normalize_caption(text):
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "\"'“”":
        body = body[1:-1].strip()
    if not body:
        raise ValueError("the model returned an empty caption")
    return body


def postprocess(text, mode, width, height):
    if mode == "Layout":
        return compile_layout(text, width, height)
    return normalize_caption(text)
