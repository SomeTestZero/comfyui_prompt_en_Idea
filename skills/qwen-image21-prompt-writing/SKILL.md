---
name: qwen-image21-prompt-writing
description: Rewrite a rough request into a Qwen-Image-2.1 prompt for text-to-image (T2I), instruction-based image editing (Edit), or a character reference sheet (CharacterSheet: multi-view turnaround with literal labels). Use when writing prompts for TextEncodeQwenImage21, addressing multi-image inputs as <image1>/<image2>, committing quoted rendered text and typography, RGBA transparent output, reference-driven scene composition, or 人设图 / model sheets.
---

# Qwen-Image 2.1 Prompt Writing

One model serves both text-to-image and instruction editing; the register differs by task. Follow the routed official guide plus the hard rules below:

- **T2I** → `references/t2i-en.txt` — the official PE-T2I system prompt ("Image Prompt Rewriting Expert"): an observer's description of the finished frame in eight ordered steps (read the brief → fix the frame → opening sentence → inventory → walk the frame → set every text string → lighting → closing sentence).
- **Edit** → `references/edit-en.txt` — the official PE-I2I system prompt ("Edit Prompt Enhancer"): a precise editing directive anchored on the input image(s), governed by attribute disentanglement.
- **CharacterSheet** → `references/charactersheet-en.txt` — the character reference sheet (人设图 / model sheet / turnaround) composition guide: one shared appearance inventory, panel-by-panel camera walk (front/side/back elevations + facial close-up), flat solid backdrop with nothing under the figures, no text on the sheet (title/labels only when the user explicitly asks), orthographic discipline; with reference images, identity is anchored to the images instead of words.

## Hard rules

1. Output exactly one continuous paragraph — the prompt text only. No bullets, no JSON, no markdown, no explanations. The guides' JSON envelope (`rewritten_prompt` / `wh_ratio` / `ratio_follow`) is replaced by plain-text output here; everything they say about the content and formatting of `rewritten_prompt` still applies.
2. Never write resolution, aspect ratio, or pixel-count words ("2K", "4K", "16:9", "2048x2048") into the prompt — separate workflow nodes set the size, and quality boasts are banned anyway.
3. Image addressing (Edit): two or more inputs MUST be referenced as `<image1>`, `<image2>`, … in slot order — never "the first image" / "第一张图". State each image's role: which one is the canvas whose composition and untargeted content survive, and which supply material to transfer. A single input is referred to naturally ("the image", "图像中") without tags.
4. Rendered text is literal: exact characters in double quotes, every readable element quoted, kept in its own script, monolingual (no bilingual pair or translation gloss unless the user asks). Text you cannot commit to character for character is not added at all.
5. Language: Edit — descriptive prose follows the user's instruction language (Chinese→Chinese, English→English, anything else→English); quoted rendered text follows the PE-I2I language decision (B). T2I — the description is always English; quoted text stays in its own script.
6. Attribute disentanglement (Edit): edit exactly the named attributes and push each to an unmistakable degree; hold everything else at input fidelity. Lock what stays with one blanket affirmative preservation clause ("the background stays exactly as in the input") — never by describing kept content in detail, never as a prohibition. Identity survives every edit unless the user targets it: faces, signature accessories, product design and markings, and the input's rendering medium. Where identity comes from a reference image, point at that image instead of describing features in words.
7. Only what was asked (Edit): no extra operations, no cleanup of unmentioned defects or clutter, no "helpful" reframing. When an edit removes, moves, or reveals something, say enough about the newly exposed region that the result stays physically coherent. Preserve creative or impossible intent rather than correcting it.
8. No empty quality boosters ("masterpiece", "best quality", "8K", "highly detailed") — every phrase must describe something visible. Qwen-Image-2.1 runs at CFG 1.0, so negative prompts do nothing: exclusions are written as positive statements of the desired state ("exactly one person appears in the frame" instead of "no extra people").
9. Transparent output: when transparency is requested (the node's `background` input set to Transparent, or the user's brief), wrap the description in the official RGBA wording verbatim: `This is an RGBA image with transparency. <description>. The image has alpha channel and the background is transparent.` and describe no backdrop of any kind.
10. If the user already wrote a well-formed prompt, lightly polish and finalize it instead of rebuilding — preserve their phrasing, order, and direction.

## Length

- T2I: about twenty sentences / 400–500 words whether the brief was three words or three hundred (official PE-T2I size); a single quiet subject runs shorter, a dense poster with much text runs longer.
- Edit: follow the PE-I2I intent branch — a local change ("this picture changed") stays a restrained directive that says exactly what changes; a "new picture of this subject" (photo shoot, poster, composite, infographic) is actively constructed to a professional standard, and elaboration scales with what was asked.
- CharacterSheet: thirty to forty sentences / 500–650 words — appearance inventory once, then the panel walk; a three-panel sheet runs shorter.

## Reference

`references/t2i-en.txt` and `references/edit-en.txt` are verbatim archives of the official system prompts shipped with [Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I) / [Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I) — the prompt-rewriting models the Qwen-Image-2.1 README recommends for best results. Their full rules, tables, and worked examples apply; where they conflict with this file on output shape, this file wins. `references/charactersheet-en.txt` is our own composition guide in the same register (character reference sheets are a layout class the official pair does not cover), not an archive.
