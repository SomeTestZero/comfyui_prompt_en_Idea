---
name: img2prompt-natural
description: Reverse-prompt an image into a ready-to-use natural-language generation prompt (Krea 2 / FLUX / modern T2I style). Single faithful English paragraph.
---

# Image → Natural-Language Prompt

Reconstruct a generation-ready prompt that would recreate the attached image in a modern natural-language text-to-image model (Krea 2, FLUX, SD3 era).

## Hard rules

1. Output exactly one English paragraph. No bullets, no section headers, no markdown.
2. Faithfulness first: describe only what is visible. Never invent identities, brands, camera models, focal lengths, or hidden objects. If a detail is genuinely ambiguous, pick the most likely reading and keep it generic rather than specific.
3. No generic praise or empty quality tags: no "masterpiece", "stunning", "best quality", "8K", "highly detailed" as filler. Every phrase must describe something visible.
4. Bind every attribute, garment, color, and prop to the correct subject. For multiple subjects use explicit clauses ("On the left... On the right... Behind them...").
5. Visible text: transcribe exactly inside double quotes, preserving capitalization and line breaks; state its position and style.
6. State the medium first (photograph, digital painting, anime key visual, 3D render, ink illustration, collage...) — this anchors the whole reconstruction.

## Coverage, in visual priority order

medium → primary subject (count, position, scale) → pose/action/gaze → appearance/clothing → camera and composition (shot size, angle, perspective, depth of field, crop) → environment (foreground/midground/background, occlusion) → lighting (source, direction, softness, shadows, highlights) → palette (dominant/secondary/accent, warm-cool) → materials and texture → style/production method → visible text.

## Length

80–220 English words for a typical image; a simple icon or texture can be shorter, a dense scene up to ~300. Dense but non-redundant; cut repeated adjectives first.
