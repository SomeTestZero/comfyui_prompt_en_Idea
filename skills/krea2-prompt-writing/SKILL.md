---
name: krea2-prompt-writing
description: Rewrite a rough idea into a production-ready Krea 2 (RAW/Turbo) text-to-image prompt, plain scene (T2I) or character reference sheet (CharacterSheet: multi-view turnaround). Natural-language single paragraph, official Krea 2 rules.
---

# Krea 2 Prompt Writing

Rewrite the user's request into one cohesive English paragraph that Krea 2 can parse cleanly. Krea 2 is a natural-language model: prefer concrete visual relations over tag lists or quality keywords.

Task mode routing: **T2I** → the official guides in `references/`; **CharacterSheet** → `references/charactersheet-en.txt`, the character reference sheet (人设图 / model sheet / turnaround) composition guide: one shared appearance inventory, panel-by-panel camera walk (front/side/back elevations + a wider facial close-up) with each panel bound to its horizontal position, a concrete studio set with one floor ring marker per standing figure, an explicit figure count against panel drift, no text on the sheet. Sheet identity comes only from stated words, reference images, or the pinned LoRA profile — a bare character name invents nothing (the character LoRA owns the look) and names/trigger terms stay verbatim in their own script.

## Hard rules

1. Output exactly one English paragraph. No bullets, no JSON, no markdown, no explanations.
2. Preserve all user-specified subjects, actions, colors, counts, and spatial relationships. Do not add objects, people, or animals the user did not imply.
3. If the user explicitly named a medium ("photo of", "illustration of", "3D render of", "anime"), honor it exactly — never pivot to another medium.
4. Visible text: wrap the exact words in double quotes and state position, size hierarchy, and font style. Preserve Chinese characters verbatim if requested.
5. Never use empty quality tags: no "masterpiece", "best quality", "8K", "highly detailed" as filler. Every phrase must describe something visible.
6. Krea 2 Turbo runs at CFG 0, so negative prompts do not work. Express exclusions as positive statements of the desired visible state:
   - bad: "no clutter" → good: "a restrained background with only three broad geometric forms"
   - bad: "no extra people" → good: "exactly one person appears in the frame"
7. If reference images are attached, ground style, palette, and subject details in what is actually visible in them; never invent appearance details. Keep the text prompt about content and let the references carry style.

## Assembly order

Use only the modules that materially affect the image; skip the rest. Default order:

medium → primary subject (count, position, scale) → pose/action/gaze → appearance/clothing → camera/composition (shot size, angle, depth of field) → environment (foreground/midground/background relations) → lighting (source + direction + softness + shadows) → color system (dominant/secondary/accent, warm-cool) → materials and surface behavior → style/texture → exact typography → positive constraints.

## Length

- Exploration or simple subject: 20–60 words, one sentence is fine.
- Controlled generation (portrait, product, poster, CG scene): 80–220 words, dense but non-redundant.
- Genuinely complex scenes: up to ~350 words. Cut repeated adjectives first.
- Character sheet: 250–350 words — appearance inventory once, then the panel walk; a three-panel sheet runs shorter.

Longer is not automatically better; choose density by intent. If the user's prompt is already detailed, lightly polish and finalize instead of heavily expanding — preserve their phrasing and direction.

## Multi-subject control

Bind every attribute to its own subject with explicit clauses ("On the left... On the right... Behind them...") so attributes never leak between similar subjects. State counts, left/right placement, occlusion, and interaction explicitly.

## Reference

See references/official-prompting.md for the official Krea 2 guidelines and twenty 2K example prompts (CG render, anime, photorealism, collage, ink, surreal) — match their phrasing register. references/expansion.txt is Krea's own LLM expansion system prompt; its faithfulness rules apply. references/charactersheet-en.txt is our own character-sheet composition guide in the same register (reference sheets are a layout class the official pair does not cover), not an archive.
