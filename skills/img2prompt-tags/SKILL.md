---
name: img2prompt-tags
description: Reverse-prompt an image into a booru-style comma-separated tag list (Danbooru vocabulary, SD1.5/Pony/SDXL era). Lowercase underscore tags only.
---

# Image → Booru Tag List

Reconstruct a Danbooru-style tag list that would recreate the attached image in tag-era models (SD1.5, Pony, SDXL booru finetunes).

## Hard rules

1. Output ONLY the tag list: lowercase, underscores for multi-word tags, comma + space separated. No sentences, no numbering, no explanations, no markdown.
2. Use established Danbooru vocabulary only (`1girl`, `solo`, `looking at viewer`, `upper body`, `depth of field`, `film grain`). Do not coin new phrases; if no established tag fits a detail, drop the detail rather than invent a tag.
3. Faithfulness: tag only what is visible. Never guess identity, franchise, artist name, or character name unless it is unambiguous (and prefer describing appearance over naming).
4. No rating tags beyond `safe`/`questionable`/`explicit` when obvious; place rating first if used.
5. Count and solo tags come first when applicable: `1girl`, `2boys`, `solo`, `multiple girls`.

## Tag order

count/solo → subject type and gender → hair (length, color, style) → eyes → expression → pose/gaze → clothing and accessories → body-relevant framing tags (`upper body`, `cowboy shot`, `full body`) → camera/composition (`from below`, `close-up`, `dutch angle`) → setting/background (`simple background`, `outdoors`, `night`) → lighting → style/medium (`photorealistic`, `anime`, `3d`, `monochrome`) → atmosphere/color (`warm colors`, `high contrast`).

## Length

25–60 tags for a typical image. Every tag must earn its place; do not pad with near-synonyms.

## Non-anime input

Booru vocabulary is anime-biased. For photographs or CG renders, keep using established general tags (`photorealistic`, `realistic`, `film grain`, `bokeh`) and accept sparser coverage rather than forcing anime-specific vocabulary onto a photo.
