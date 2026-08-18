import random
import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, log, make_progress_cb
from .llm_local import LocalLLM, tensor_to_base64_jpeg
from .nodes_local import advanced_model_inputs, build_model_config, derive_mode, model_input, sampling_inputs

IDEA_SYSTEM = """You are a screenwriter for short AI-generated videos. Write a story draft from the user's request.

Rules:
- The user's request defines the story: its subject, style, pacing, shot count, and any explicit requirements are mandatory - build everything around them. Whatever the user left open, invent freely and concretely (characters, actions, scene details).
- The draft is the only thing the downstream prompt writer ever sees - the user's original words never reach it. Restate every explicit requirement from the request or the given ingredients (style/genre, shot policy such as a single continuous take with no cuts, pacing, mood, anything insisted on) in the opening line of the draft, in the user's own words plus the standard English production term in parentheses when one exists, e.g. "风格：动漫（2D-animated）；镜头：单镜头一镜到底（one continuous take, no cuts），只运镜". A requirement not written in the draft is lost.
- Format: the opening line (explicit requirements if any, then the setup: subject / setting / tone), then the story as shot beats. Default to a shot-by-shot breakdown with framing, the key action, and any spoken lines in quotes - unless the user asked otherwise (e.g. a single long take).
- Fit the {duration}-second runtime: everything in it — actions, shot changes, spoken lines — must be playable within {duration} seconds at a natural pace.{image_note}
- Write in {language}.
- Output ONLY the story draft. No title, no commentary, no markdown headers."""

MODE_GUIDANCE = {
    "Ref2VA": "The attached reference image(s) show the main character/subject. Keep their visible look consistent, but the place, event, and plot are yours to invent — go far beyond what the image shows.",
    "FL2VA": "The attached images are the video's opening and closing keyframes. Invent the story that happens BETWEEN them: the first shot starts from the first frame's scene, the last shot lands on the last frame's scene, and the middle is a real development, not a straight line.",
    "I2VA": "The attached image is the video's opening keyframe. Its subject and setting are only the starting point — invent what happens next, beyond anything visible in the image.",
    "L2VA": "The attached image is the video's closing keyframe. Invent the story that leads up to this final scene.",
}

# No output cap: llama.cpp treats max_tokens <= 0 as "fill the context", and
# the system prompt bounds the length; the model stops at EOS.
IDEA_MAX_TOKENS = -1

# Random-mode ingredient pools. The seed picks the combination, the model only
# weaves them — this keeps "random" actually random instead of collapsing onto
# the model's few favorite tropes. With images connected, subject/setting come
# from the pictures, so only TWISTS and MOODS are drawn.
GENRES = [
    "科幻", "奇幻", "日常治愈", "悬疑", "冒险", "轻喜剧", "自然纪录片", "历史古装", "赛博朋克", "童话",
    "都市传说", "太空歌剧", "末世废土", "武侠", "蒸汽朋克", "海洋探险", "微观世界", "怪谈", "美食纪录", "时间循环",
]
SUBJECTS = [
    "一只流浪猫", "退休的灯塔看守人", "送外卖的机器人", "卖花的老奶奶", "失眠的天文台研究员",
    "会修表的小狐狸", "深夜食堂老板", "实习小巫师", "古董店掌柜", "地铁站务员",
    "山区邮递员", "海底观测站工程师", "木偶戏艺人", "图书管理员", "夜班出租车司机",
    "守林人", "马戏团小丑", "渔村少年", "AI 管家", "云朵牧羊人",
]
SETTINGS = [
    "废弃的温室花房", "凌晨四点的夜市", "山顶缆车终点站", "老式火车卧铺车厢", "下雨的城中村天台",
    "极光下的冰原", "深海热泉旁", "沙漠中的绿洲小镇", "百年图书馆的禁书区", "漂浮的空中岛屿",
    "地下溶洞暗河", "台风来临前的海边", "雪夜的温泉旅馆", "满是藤蔓的旧游乐园", "空间站观景舱",
    "黄昏的麦田", "凌晨的机场候机厅", "梅雨季节的老巷", "火山脚下的村庄", "无边无际的盐沼",
]
TWISTS = [
    "捡到一台还能放映的老胶片机", "收到一封寄给十年后的信", "发现门后是一片海", "所有钟表同时倒着走",
    "一只会说话的鸟带来口信", "下了一场带着光点的雪", "找到一张会动的旧照片", "听到墙里传出音乐声",
    "影子突然比自己先动了一步", "挖到一颗会发光的石头", "雨停后城市变成了微缩模型", "月亮近得能看清环形山",
    "电梯停在了不存在的楼层", "所有植物一夜之间开花了", "捡到一本写着自己名字的日记", "海雾中驶来一艘旧帆船",
    "屋顶上落了一颗小星星", "冰箱里住着一个小小的冬天", "每天同一时间出现的陌生访客", "地图上多出一个没有名字的小镇",
]
MOODS = [
    "温暖治愈", "紧张刺激", "荒诞幽默", "宁静悠远", "神秘莫测",
    "热血沸腾", "淡淡忧伤", "史诗恢宏", "诡谲奇异", "轻松欢快",
]


class H3IdeaGeneratorLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3IdeaGeneratorLocal",
            display_name="H3 Idea Generator (Local GGUF) 灵感生成",
            category="prompt",
            description="Generate a short video story draft (setup + shot-by-shot beats) with a local GGUF model, to feed the H3 Prompt Enhancer's prompt input when you're out of inspiration. Wire the same keyframes/reference images as the enhancer and the story is written around them: a plot that travels from first to last frame, or one starring the reference subject. Empty hint + no images = the seed picks story ingredients and the model weaves them, so every seed gives a different but reproducible story. The story is sized to the duration input and the draft is stamped with it, so the enhancer schedules shots within the runtime. Explicit requirements from the hint (style, shot policy, ...) are restated in the draft's opening line with their English production terms, so the enhancer keeps them. Image input needs a *mmproj*.gguf next to the model.",
            inputs=[
                io.String.Input("hint", multiline=True, default="", tooltip="Optional theme/clue to riff on — no need to write the duration here, it has its own input. Leave empty for a random story draft from seed-picked ingredients (with images: only plot-twist and mood are drawn, subject/setting come from the pictures)."),
                io.Combo.Input("language", options=["中文", "English"], default="中文", tooltip="Language of the generated story; the enhancer accepts either."),
                model_input(),
                io.Combo.Input("thinking", options=["disabled", "enabled"], default="disabled", advanced=True),
                io.Float.Input("temperature", default=-1.0, min=-1.0, max=2.0, step=0.05, tooltip="-1 = auto: follow the Qwen3.8 official preset for the thinking mode (1.0 thinking / 0.7 non-thinking). >=0 = manual override."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input("keep_loaded", default=False, tooltip="Off: unload the model after generation so later nodes get the VRAM back. On: keep it resident for repeated runs."),
                *advanced_model_inputs(),
                io.Image.Input("first_frame", optional=True, tooltip="First keyframe image; the story starts from this scene (with last_frame: ends at that scene). Same wiring as the enhancer."),
                io.Image.Input("last_frame", optional=True, tooltip="Last keyframe image; the story builds up to this scene."),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("ref_image", tooltip="Subject/character reference; the story stars what is visible in it. ref_image_N matches the enhancer's socket of the same name."),
                        prefix="ref_image_", min=0, max=9,
                    ),
                    tooltip="Reference images (characters, style, props); socket names mirror the H3 Prompt Enhancer / MiniMax H3 Reference to Video nodes — wire the same image to the same-numbered ref_image_N on all of them.",
                ),
                # Appended last: saved workflows store widget values positionally.
                io.Float.Input("duration", default=10.0, min=4.0, max=15.0, step=0.5, tooltip="Target video duration in seconds (H3 supports 4-15) — wire the same duration that feeds the sampler's frame count. The story is sized to fit it, and the value is stamped on the draft so the enhancer schedules shots within the runtime."),
                *sampling_inputs(),
            ],
            outputs=[io.String.Output(display_name="idea")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, hint, language, model, thinking, temperature, seed,
                keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe, first_frame=None, last_frame=None,
                reference_images=None, duration=10.0,
                top_p=-1.0, top_k=-1, presence_penalty=-1.0, reasoning_effort="low"):
        mode = derive_mode(first_frame, last_frame, reference_images)
        dur = f"{duration:g}"

        # (tensor, role) pairs in <Picture N> order, same convention as the enhancer
        frames = []
        if reference_images:
            for name in sorted(reference_images, key=lambda n: int(n.rsplit("_", 1)[-1])):
                for frame in reference_images[name]:
                    frames.append((frame, "reference"))
        if first_frame is not None:
            frames.append((first_frame[0], "first frame"))
        if last_frame is not None:
            frames.append((last_frame[0], "last frame"))

        hint = hint.strip()
        parts = []
        if frames:
            listing = ", ".join(f"<Picture {i + 1}> ({role})" for i, (_, role) in enumerate(frames))
            parts.append(f"{len(frames)} image(s) attached in order: {listing}.")
        if hint:
            parts.append(f"用户要求：{hint}\n按上面的要求编一个故事草稿，把细节补充具体；要求里没说到的部分自由发挥。草稿开头行需原样列出上面的明确要求，并附对应的英文术语。")
        else:
            rng = random.Random(seed)
            if frames:
                # subject/setting come from the pictures; the seed only picks what happens
                ingredients = f"意外元素：{rng.choice(TWISTS)}；基调：{rng.choice(MOODS)}"
                parts.append(f"随机元素：{ingredients}\n以画面中主体为主角，让「意外元素」在故事里真实发生并推动转折，基调决定整体氛围。不要复述画面内容。")
            else:
                ingredients = (
                    f"类型：{rng.choice(GENRES)}；主角：{rng.choice(SUBJECTS)}；场景：{rng.choice(SETTINGS)}；"
                    f"意外元素：{rng.choice(TWISTS)}；基调：{rng.choice(MOODS)}"
                )
                parts.append(f"随机元素：{ingredients}\n把这些元素编织成一个有转折的故事草稿，按分镜展开。")
            log(f"idea ingredients (seed={seed}): {ingredients}")
        user_text = "\n".join(parts)

        image_note = ""
        if frames:
            image_note = f"\n- {MODE_GUIDANCE[mode]}"

        config = build_model_config(model, thinking, n_ctx, n_gpu_layers, n_cpu_moe, reasoning_effort)
        sampling = LocalLLM.resolve_sampling(thinking == "enabled", temperature, top_p, top_k, presence_penalty)
        try:
            LocalLLM.load(config)
            if frames and not LocalLLM.has_vision():
                raise ValueError(f"{len(frames)} image(s) connected, but no *mmproj*.gguf was found next to {model}; image input needs a vision projector.")

            user_content = [{"type": "text", "text": user_text}]
            for frame, _ in frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{tensor_to_base64_jpeg(frame)}"},
                })
            messages = [
                {"role": "system", "content": IDEA_SYSTEM.format(image_note=image_note, language=language, duration=dur)},
                {"role": "user", "content": user_content},
            ]
            log(f"idea gen: mode={mode} model={model} seed={seed} lang={language} duration={dur}s images={len(frames)} hint={hint[:100]!r}")
            log(f"user prompt: {user_text[:500]}")
            content = LocalLLM.generate(messages, max_tokens=IDEA_MAX_TOKENS, sampling=sampling,
                                        seed=seed, on_text=make_progress_cb(cls.hidden.unique_id))
            # Stamp the runtime on the draft so the downstream enhancer can
            # schedule shot timings against the requested duration.
            header = f"总时长：{dur} 秒" if language == "中文" else f"Total runtime: {dur} seconds"
            content = f"{header}\n\n{content}"
            log(f"idea ({len(content)} chars):\n{content[:500]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "idea",
                "task_type": f"local/idea/{mode}",
                "model": model,
                "seed": seed,
                "duration": duration,
                "input": hint or (f"random, {len(frames)} image(s)" if frames else "random"),
                "output": content,
            })
            return io.NodeOutput(content)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if keep_loaded:
                LocalLLM.clear_context()
            else:
                LocalLLM.unload()
