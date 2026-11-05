# MOSS-Image Prompt Writing

Turn a rough image idea into a prompt for MOSS-Image (OpenMOSS: 5B single-stream
DiT, FLUX.2 small VAE, Qwen3.5-9B text encoder). Two deliverables: a finished-image
**caption**, or a **layout draft** with bounding boxes. This skill never generates
images and never renders text itself.

Adapted from the official
[`moss-image-prompt-enhancer`](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer)
skill (Apache-2.0), with the filesystem/compile-tool steps removed: the node
compiles the layout draft and validates the canvas.

## Choose the deliverable

Honour an explicit request for a caption or a layout. Otherwise use a caption for
ordinary scenes, photographs and illustrations. Use a layout when specific text
placement, a poster, several panels, or controlled object relationships benefit
from bounding boxes. A detailed image does not automatically need a layout.

Preserve the user's subjects, counts, relationships, medium, style, exclusions and
intended visible wording. Resolve ordinary creative gaps yourself; do not invent
factual event logistics or specifications as if the user supplied them, and do not
turn a minimalist brief into a generic title/hero/footer template.

Default to English descriptive prose; keep any requested output language. Preserve
visible text verbatim in its original language, including punctuation,
capitalisation and requested line breaks. A quoted sentence in an instruction is
not necessarily text to render: distinguish visible wording from reference
material and constraints.

## Develop the image

Describe visible choices: subject and action, relevant appearance, setting,
camera/framing, medium, lighting, materials, palette and mood. Add detail where it
makes the requested image more specific. Avoid redundant quality tags, unrequested
style changes and invisible process notes. Keep explanations of JSON, bounding-box
annotations and conversion steps out of image-description fields; those are tool
instructions, not depicted content.

For designed images, compose the information and imagery together: reading order,
relative prominence, coherent groups and text legible at the chosen pixels. Write
the intended copy, never placeholders such as "headline here". Do not import
unrelated names, watermarks or copy from a reference.

## Caption

Write one cohesive caption. Spatial relations go in words; any visible wording is
quoted exactly. It may contain paragraphs. Do not add coordinates, lists, headings
or JSON-layout fields to a caption just to make it look structured. Do not describe
watermarks or artist signatures.

MOSS-Image renders line breaks inside quoted text when they are explicit: use a
literal newline or `\n` inside the quote when the requested design needs it.

## Layout

Read `references/layout-en.txt` before writing a layout draft. Use readable JSON
with `width`, `height`, a flat `items` array, `bbox_coord_space="abs"` and
`bbox_format="xyxy"`.

Draft boxes use `[x_min, y_min, x_max, y_max]` in actual pixels from the canvas
top-left. Keep x in `0..width` and y in `0..height`. Never use normalised
coordinates, and never center, reorder or quantise the boxes yourself: the node
converts them to the model's `abs_centralized`/`yxyx` form.

Honour supplied dimensions. MOSS-Image dimensions are positive multiples of 16;
the canvas given by the node is already valid and must be used unchanged. Use
boxes for coherent visible units, splitting only when independent placement
matters. Natural occlusion and overlap are allowed. A box is a placement
annotation, not an instruction to draw a border, panel or sticker. Describe
appearance in captions and global treatment in `summary.caption`; let the boxes
agree with any spatial descriptions. Keep exact visible strings in text items'
`text` fields.

## Output contract

- Caption mode: reply with the caption only — plain prose, no title, introduction,
  explanation, surrounding quotes, markdown or JSON.
- Layout mode: reply with exactly one JSON object in the draft schema. No prose,
  no markdown fences, no comments.
- Never output two alternatives or a self-critique.
