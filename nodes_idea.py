import random
import re
import traceback
from datetime import datetime

from comfy_api.latest import io

from .common import append_history, log, make_progress_cb
from .llm_local import LocalLLM, local_options, tensor_to_base64_jpeg
from .nodes_local import advanced_model_inputs, build_model_config, derive_mode, model_input, sampling_inputs

IDEA_SYSTEM = """You are a screenwriter for short AI-generated videos. Write a story draft from the user's request.

Rules:
- The user's request is binding - subject, style, pacing, shot count, any explicit requirement. Everything left open is yours to invent, freely and concretely.
- The draft is the only thing the downstream prompt writer ever sees. Open with one line that restates every explicit requirement in the user's words plus the standard English production term in parentheses, e.g. "风格：动漫（2D-animated）；镜头：单镜头一镜到底（one continuous take, no cuts）", then the setup: subject / setting / tone. A requirement not written in the draft is lost.
- Then the story as filmable beats - framing, the chain of concrete actions (several consecutive actions, not just one), any spoken lines in quotes. The actions of a beat must spread evenly across its full duration - the final second is still mid-motion, never winding down. Number beats only where a real cut happens; a single continuous take stays one unnumbered narration, because 镜头1、镜头2 numbering reads as cuts downstream.
- Everything - actions, cuts, spoken lines - must be playable within {duration} seconds.{image_note}{segment_note}
- Write in {language}. Output ONLY the story draft: no title, no commentary, no markdown."""

SEGMENT_NOTE = """
- Exception to the unnumbered-narration rule above: this single continuous take is generated as {total_segments} chained segments forming one continuous cause-and-effect story arc - each beat is a time slice of the same unfolding event, not a separate skit. After the setup, write exactly {total_segments} beats, each opening with its own label line "段1", "段2", ... ("Segment 1", "Segment 2", ... in English). {pacing} A beat covers only that segment's action and must continue seamlessly from the previous beat's ending state - no cuts, no new framing, no re-introductions. Each beat must carry enough events to honestly fill its screen time: scale the action chain to the beat's own seconds - roughly one concrete, filmable action or reaction every 1-1.5 seconds - and keep a natural ebb and flow: intense bursts alternate with brief, still-moving transitions (repositioning, bracing, circling), so the pacing breathes instead of staying flat; never pad the seconds with slow motion, freezes, lingering gazes, or idle scenery, and keep the final second mid-motion. The camera is one unbroken eye: every beat inherits the previous beat's camera state (same position, distance and angle) and only continues it with smooth movement - follow, orbit, handheld drift; no jumps, no re-framing openings. Each beat ends with its own hand-off line "段末：" ("End of segment:" in English) stating the exact on-screen state at the final frame: subjects' poses, positions, motion direction, and the current camera framing/movement - the next segment starts from exactly this state. The labels are parsed mechanically downstream, so they must be exact."""

MODE_GUIDANCE = {
    "Ref2VA": "The attached reference image(s) show the main character/subject. Keep their visible look consistent, but the place, event, and plot are yours to invent — go far beyond what the image shows.",
    "FL2VA": "The attached images are the video's opening and closing keyframes. Invent the story that happens BETWEEN them: the first shot starts from the first frame's scene, the last shot lands on the last frame's scene, and the middle is a real development, not a straight line.",
    "I2VA": "The attached image is the video's opening keyframe. Its subject and setting are only the starting point — invent what happens next, beyond anything visible in the image.",
    "L2VA": "The attached image is the video's closing keyframe. Invent the story that leads up to this final scene.",
}

# No output cap: llama.cpp treats max_tokens <= 0 as "fill the context", and the system
# prompt bounds the length; the model stops at EOS. config.json "local".max_tokens can
# cap it for batch runs (see README).

# Seed-drawn ingredient pools, used on every run - hint or not. The seed picks
# the combination, the model only weaves them - this keeps "random" actually
# random instead of collapsing onto the model's few favorite tropes (a style-only
# hint like "3DCG" leaves the subject to the model, and every seed then lands on
# the same translucent-glowing-mechanical creature). Variety must come from
# these inputs, never from directives in the prompt: any instruction about story
# rhythm ("must have a twist", "keep it calm") biases every draft toward one
# shape. EVENTS mixes quiet, everyday, humorous, and a few surreal happenings so
# the story always has a core to hang on. With images connected, subject/setting
# come from the pictures, so only EVENTS and MOODS are drawn.
GENRES = [
    "科幻", "奇幻", "日常治愈", "悬疑", "冒险", "轻喜剧", "自然纪录片", "历史古装", "赛博朋克", "童话",
    "都市传说", "太空歌剧", "末世废土", "武侠", "蒸汽朋克", "海洋探险", "微观世界", "怪谈", "美食纪录", "时间循环",
    "水墨动画", "黏土定格动画", "复古胶片", "黑色侦探", "体育竞技", "歌舞音乐剧", "极地纪实", "市井烟火",
    "灾难片", "公路片", "校园", "怪兽片", "克苏鲁", "太阳朋克", "谍战", "荒诞喜剧", "梦境", "伪纪录片",
    "默片喜剧", "剪纸动画", "皮影戏", "宫廷", "仙侠", "民间故事",
]
SUBJECTS = [
    "一只流浪猫", "退休的灯塔看守人", "送外卖的机器人", "卖花的老奶奶", "失眠的天文台研究员",
    "会修表的小狐狸", "深夜食堂老板", "实习小巫师", "古董店掌柜", "地铁站务员",
    "山区邮递员", "海底观测站工程师", "木偶戏艺人", "图书管理员", "夜班出租车司机",
    "守林人", "马戏团小丑", "渔村少年", "AI 管家", "云朵牧羊人",
    "外卖骑手", "退休消防员", "夜市烧烤摊主", "小学自然课老师", "考古队学徒",
    "修船的老船匠", "第一次进城的小镇青年", "盲人调音师", "便利店夜班店员", "观光热气球驾驶员",
    "流浪歌手", "开煎饼摊的前程序员", "铁路巡道工", "游乐园人偶扮演者", "养蜂人",
    "晒盐工", "修伞匠", "捏面人的手艺人", "皮影戏班主", "一只成精的橘猫",
    "成了精的自动售货机", "一盏怕黑的路灯", "退休的门神", "实习土地公", "孟婆汤铺的伙计",
    "给月亮换灯泡的人", "贩卖梦境的小贩", "许愿池里的老龟", "商场门口的充气摇摆人", "流浪的扫地机器人",
    "被遗忘的充气恐龙", "一头独行的鲸", "迁徙途中的信天翁", "密室逃脱NPC演员", "酒店试睡员",
    "深夜电台主播", "老裁缝", "报刊亭主人", "街头最后一座电话亭",
]
SETTINGS = [
    "废弃的温室花房", "凌晨四点的夜市", "山顶缆车终点站", "老式火车卧铺车厢", "下雨的城中村天台",
    "极光下的冰原", "深海热泉旁", "沙漠中的绿洲小镇", "百年图书馆的禁书区", "漂浮的空中岛屿",
    "地下溶洞暗河", "台风来临前的海边", "雪夜的温泉旅馆", "满是藤蔓的旧游乐园", "空间站观景舱",
    "黄昏的麦田", "凌晨的机场候机厅", "梅雨季节的老巷", "火山脚下的村庄", "无边无际的盐沼",
    "深夜自习室", "郊外废弃汽车影院", "老城区理发店", "山谷悬索桥", "清晨的渔港码头",
    "屋顶菜园", "冬夜的火车站台", "巷尾的旧邮局",
    "云雾缭绕的深山村落", "深秋的无人林场", "大雪封山的林间木屋", "海边的悬崖灯塔", "山顶的天文台",
    "末班车后的地铁站台", "城市地下的换乘通道", "近海钻井平台", "街角的24小时便利店",
    "凌晨的水产批发市场", "深夜的自助洗衣房", "城郊的废品收购站", "老小区的健身器材区", "防空洞改建的菜市场",
    "胡同深处的公共澡堂", "沙漠公路的唯一加油站", "极地考察站的厨房", "废弃的矿坑小镇", "被淹一半的地下车库",
    "未完工的摩天楼顶层", "火箭发射场观景台", "冰川裂缝边缘", "巨型树的树冠层", "蘑菇森林深处",
    "鲸鱼的背上", "时钟内部的齿轮间", "无限延伸的自动扶梯", "镜子里的世界", "沉没海底的古城",
    "瀑布后的水帘洞", "千年银杏树下", "油菜花海的田埂", "老茶馆", "雾中的渡口",
]
EVENTS = [
    "泡好的茶刚好在雨停时端上桌", "修好的旧钟表重新走动起来", "窗外雪停，月光第一次照进屋里",
    "老人把最后一枚糖递给小孩", "猫跳上膝头打起盹来", "刚出炉的面包香气引来邻居",
    "孩子第一次把风筝放上了天", "远处的灯塔亮起来了", "候鸟群恰好从头顶飞过",
    "风把一张旧车票吹到脚边", "旧收音机忽然放出一段熟悉的旋律", "一封没有署名的信被送到门口",
    "墨水在纸上晕开成一个形状", "帽子被风吹走，落在一个陌生人头上", "鸽子叼走了三明治的一角",
    "机器人把咖啡端反了方向", "影子突然比自己先动了一步", "月亮近得能看清环形山",
    "雨停后城市变成了微缩模型", "所有钟表同时倒着走",
    "停电的夜里全楼的人下楼看星星", "修了很久的灯串第一次全部亮起", "候车时身边的陌生人分来一只耳机",
    "拖鞋被浪卷走又自己漂了回来", "鹦鹉学会了门铃声骗开了门", "旧地图上标着的小巷真的存在",
    "雨后水洼里倒映出整条街",
    "第一缕阳光正好落在课桌上", "楼下的桂花一夜之间全开了", "晾衣绳上的床单鼓成了船帆",
    "蒲扇摇着摇着天就黑了", "有人把走调的钢琴弹顺了", "腌了一冬的酸菜开坛了",
    "冰棍掰成两半，一人一半", "火柴在风里第三次才划着", "仙人掌在夜里开了一朵花",
    "两个人的伞碰在一起，拿错了", "天上的云排成了一列火车", "路灯一盏接一盏地让出光",
    "鱼缸里的鱼开始数人", "月亮掉进井里，捞了半天", "电梯多出一个不存在的楼层",
    "自动售货机掉出一张寻人启事", "雨点在半空停了一秒", "雪落在掌心没有化，变成一颗糖",
    "风筝断了线，却越飞越近", "半夜的冰箱嗡嗡声哼成了小调", "旧照片里有人眨了下眼",
    "回音晚到了整整一分钟", "一颗纽扣滚过整条街，回到原来的脚下", "台阶数着数着多出一级",
    "烟花熄灭后，天上留了一扇窗", "拨错的电话那头有人唱起了歌", "寻人启事找的是贴启事的人",
    "雨伞收起来时抖落了一地星星", "过期的日历撕到今天，日期对上了", "一只塑料袋飞得像风筝一样高",
]
MOODS = [
    "温暖治愈", "紧张刺激", "荒诞幽默", "宁静悠远", "神秘莫测",
    "热血沸腾", "淡淡忧伤", "史诗恢宏", "诡谲奇异", "轻松欢快",
    "怀旧", "好奇雀跃", "庄严肃穆", "孤独疏离",
    "迷离恍惚", "壮阔苍凉", "滑稽无厘头", "诗意隽永", "毛骨悚然", "空灵", "微甜", "暗流涌动",
]

# Lane pools. A few ingredients contradict whole categories of partners
# (micro-world vs human-scale subjects, period genres vs modern everyday life,
# a place-bound subject vs an unrelated setting), and the weaving model
# resolves such clashes by paying them lip service instead of integrating
# them. Drawing one of these switches the affected pool to a matching subset,
# so a seed's combination stays varied but can't contradict itself.
PERIOD_GENRES = {"历史古装", "武侠", "水墨动画", "宫廷", "仙侠", "民间故事"}
MICRO_SUBJECTS = [
    "一只搬运面包屑的工蚁", "一只迷路的小瓢虫", "一粒刚发芽的绿豆", "一只补网的园蛛", "玻璃罐里的萤火虫",
    "一只刚蜕壳的蝉", "一片落在水洼里的银杏叶", "墙缝里的一株蒲公英", "一只拖着露珠的蜗牛", "一只守着糖罐的灶马",
    "一颗滚落桌底的玻璃弹珠", "一只刚学会跳的蚂蚱", "一小撮被风吹散的蒲公英种子", "一只断了触角的天牛", "沉入杯底的最后一粒茶叶",
    "一只扛着花瓣的切叶蚁", "一只湿了翅膀的粉蝶", "半颗在口袋里融化的水果糖", "躲在琴凳下的一枚硬币",
]
PERIOD_SUBJECTS = [
    "赶考路过的年轻书生", "下山化缘的小和尚", "渡口撑船的老艄公", "守城打更的更夫", "绸缎庄的年轻掌柜",
    "行走乡野的卖药郎中", "深宫里的点灯人", "铸剑铺的学徒", "田间站岗的稻草人", "镖局的末席镖师",
    "绣楼里的小绣娘", "卖炊饼的矮个子", "戏班子的刀马旦", "走西口的货郎", "剃头挑子的老师傅",
    "替人写信的老先生", "城墙上晒被子的老太太", "放羊的牧童", "偷学武功的烧火丫头", "赶夜路的赶尸匠",
    "给皇陵刻碑的石匠",
]
PERIOD_SETTINGS = [
    "极光下的冰原", "沙漠中的绿洲小镇", "漂浮的空中岛屿", "地下溶洞暗河", "台风来临前的海边",
    "黄昏的麦田", "梅雨季节的老巷", "火山脚下的村庄", "无边无际的盐沼", "山谷悬索桥",
    "清晨的渔港码头", "云雾缭绕的深山村落", "深秋的无人林场", "大雪封山的林间木屋", "山顶的天文台",
    "雪夜的山间客栈", "上元灯会的桥头",
    "雾中的渡口", "瀑布后的水帘洞", "千年银杏树下", "老茶馆", "油菜花海的田埂",
    "沉没海底的古城", "挂满红灯笼的长街", "边塞的孤城", "传出晨读声的书院",
]
PERIOD_EVENTS = [
    "泡好的茶刚好在雨停时端上桌", "窗外雪停，月光第一次照进屋里", "老人把最后一枚糖递给小孩",
    "猫跳上膝头打起盹来", "孩子第一次把风筝放上了天", "候鸟群恰好从头顶飞过",
    "一封没有署名的信被送到门口", "墨水在纸上晕开成一个形状", "帽子被风吹走，落在一个陌生人头上",
    "影子突然比自己先动了一步", "拖鞋被浪卷走又自己漂了回来", "旧地图上标着的小巷真的存在",
    "雨后水洼里倒映出整条街", "更鼓声惊起了满城栖鸦", "河灯顺水流过整座城", "城门口贴出了新的告示",
    "孔明灯挂在了城楼角上", "花轿抬错了门", "庙会的糖画转出了一条龙", "更夫敲错了更，全城早起了一个时辰",
    "放榜那天下起了雨", "水井里映出两轮月亮",
]
SUBJECT_SETTINGS = {
    "退休的灯塔看守人": ["台风来临前的海边", "清晨的渔港码头", "海边的悬崖灯塔"],
    "失眠的天文台研究员": ["山顶的天文台", "空间站观景舱", "极光下的冰原"],
    "深夜食堂老板": ["凌晨四点的夜市", "梅雨季节的老巷", "雪夜的温泉旅馆"],
    "山区邮递员": ["山谷悬索桥", "火山脚下的村庄", "山顶缆车终点站", "雪夜的温泉旅馆", "云雾缭绕的深山村落"],
    "海底观测站工程师": ["深海热泉旁", "近海钻井平台"],
    "守林人": ["深秋的无人林场", "大雪封山的林间木屋", "云雾缭绕的深山村落"],
    "渔村少年": ["清晨的渔港码头", "台风来临前的海边", "海边的悬崖灯塔"],
    "修船的老船匠": ["清晨的渔港码头", "台风来临前的海边"],
    "观光热气球驾驶员": ["黄昏的麦田", "无边无际的盐沼", "极光下的冰原", "沙漠中的绿洲小镇"],
    "云朵牧羊人": ["漂浮的空中岛屿", "黄昏的麦田", "无边无际的盐沼", "极光下的冰原"],
    "地铁站务员": ["末班车后的地铁站台", "城市地下的换乘通道"],
    "夜班出租车司机": ["凌晨的机场候机厅", "郊外废弃汽车影院", "冬夜的火车站台"],
    "便利店夜班店员": ["街角的24小时便利店", "凌晨四点的夜市", "下雨的城中村天台"],
    "夜市烧烤摊主": ["凌晨四点的夜市", "下雨的城中村天台", "梅雨季节的老巷"],
    "考古队学徒": ["沙漠中的绿洲小镇", "火山脚下的村庄", "地下溶洞暗河", "百年图书馆的禁书区"],
    "图书管理员": ["百年图书馆的禁书区", "深夜自习室"],
    "第一次进城的小镇青年": ["凌晨四点的夜市", "下雨的城中村天台", "冬夜的火车站台", "凌晨的机场候机厅", "城市地下的换乘通道"],
    "开煎饼摊的前程序员": ["凌晨四点的夜市", "下雨的城中村天台", "梅雨季节的老巷"],
    "铁路巡道工": ["老式火车卧铺车厢", "冬夜的火车站台"],
    "养蜂人": ["黄昏的麦田", "云雾缭绕的深山村落", "深秋的无人林场"],
    "晒盐工": ["无边无际的盐沼", "台风来临前的海边"],
    "一头独行的鲸": ["深海热泉旁", "台风来临前的海边", "极光下的冰原"],
}


def draw_ingredients(rng):
    genre = rng.choice(GENRES)
    if genre == "微观世界":
        subject = rng.choice(MICRO_SUBJECTS)
        setting_pool, event_pool = SETTINGS, EVENTS
    elif genre in PERIOD_GENRES:
        subject = rng.choice(PERIOD_SUBJECTS)
        setting_pool, event_pool = PERIOD_SETTINGS, PERIOD_EVENTS
    else:
        subject = rng.choice(SUBJECTS)
        setting_pool, event_pool = SUBJECT_SETTINGS.get(subject, SETTINGS), EVENTS
    return genre, subject, rng.choice(setting_pool), rng.choice(event_pool), rng.choice(MOODS)


def build_idea_instruction(hint, frames, seed):
    """User text for one draft: seed-drawn ingredients plus the binding-hint
    instruction. The same (hint, frames, seed) always yields the same text; with
    images connected, subject/setting come from the pictures and the seed only
    draws event and mood."""
    hint = hint.strip()
    rng = random.Random(seed)
    parts = []
    if frames:
        listing = ", ".join(f"<Picture {i + 1}> ({role})" for i, (_, role) in enumerate(frames))
        parts.append(f"{len(frames)} image(s) attached in order: {listing}.")
        # subject/setting come from the pictures; the seed only picks what happens
        ingredients = f"核心事件：{rng.choice(EVENTS)}；基调：{rng.choice(MOODS)}"
    else:
        genre, subject, setting, event, mood = draw_ingredients(rng)
        ingredients = f"类型：{genre}；主角：{subject}；场景：{setting}；核心事件：{event}；基调：{mood}"
    log(f"idea ingredients (seed={seed}): {ingredients}")
    # A hint outranks the drawn ingredients: it is the binding request, they
    # only fill the slots it leaves open, and any that clash with it are
    # dropped outright - so "3DCG" gets a different subject/setting/event
    # every seed, while a detailed hint behaves as if alone.
    if hint and frames:
        parts.append(f"用户要求：{hint}\n随机元素：{ingredients}\n以用户要求为最高优先级、以画面中的主体为主角编一个故事草稿：要求中的每一项都必须满足；随机元素只用于补足要求没有提到的方面，与要求或画面冲突的直接弃用，没被弃用的「核心事件」必须真正发生并成为核心；基调决定整体氛围，不要复述画面内容。开头行原样列出用户的明确要求，并附英文术语。")
    elif hint:
        parts.append(f"用户要求：{hint}\n随机元素：{ingredients}\n以用户要求为最高优先级编一个故事草稿：要求中的每一项都必须满足；随机元素只用于补足要求没有提到的方面（主角、场景、事件、基调都可从中取材），与要求冲突或重复的直接弃用，其余照常融入故事，并让「核心事件」真正发生、成为核心。开头行原样列出用户的明确要求，并附英文术语。")
    elif frames:
        parts.append(f"随机元素：{ingredients}\n以画面中的主体为主角，让「核心事件」在故事里发生并成为核心（表面细节可按画面情境改编，但事件本身必须真正发生）；基调决定整体氛围，不要复述画面内容。开头行列出基调，并附英文术语。")
    else:
        parts.append(f"随机元素：{ingredients}\n用这些元素编一个故事草稿，让「核心事件」在故事里发生并成为核心。五个元素本身不可替换；若表面细节相互冲突（如场景没有车站，而事件写着「候车」），改编细节使其自洽（候车改为等船），保留元素的核心。开头行列出类型与基调，并附英文术语。")
    return "\n".join(parts)

def collect_idea_frames(first_frame, last_frame, reference_images):
    """(tensor, role) pairs in <Picture N> order, same convention as the enhancer."""
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
    return frames

def build_segment_note(segment_seconds, total_segments, beat_durations):
    """Per-segment beat-sheet appendix for the idea system prompt, or '' for single-clip drafts."""
    beat_durations = beat_durations.strip()
    if total_segments > 1 and (segment_seconds > 0 or beat_durations):
        if beat_durations:
            pacing = (f"The beats advance the story by {beat_durations} seconds respectively "
                      "(every segment after the first re-shows the previous segment's tail as pinned context, "
                      "so its new story time is shorter than the clip length; a very short last beat is a brief "
                      "closing gesture, not a full action).")
        else:
            pacing = f"Each segment runs about {segment_seconds:g}s."
        return SEGMENT_NOTE.format(total_segments=total_segments, pacing=pacing)
    return ""


class H3IdeaGeneratorLocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3IdeaGeneratorLocal",
            display_name="H3 Idea Generator (Local GGUF) 灵感生成",
            category="prompt",
            description="Generate a short video story draft (setup + shot-by-shot beats) with a local GGUF model, to feed the H3 Prompt Enhancer's prompt input when you're out of inspiration. Wire the same keyframes/reference images as the enhancer and the story is written around them: a plot that travels from first to last frame, or one starring the reference subject. The seed always picks story ingredients and the model weaves them, so every seed gives a different but reproducible story; a hint outranks the drawn ingredients - it only fills what the hint leaves open (with images: only core event and mood are drawn, subject/setting come from the pictures). Explicit requirements from the hint (style, shot policy, ...) are restated in the draft's opening line with their English production terms, so the enhancer keeps them. The story is sized to the duration input and the draft is stamped with it, so the enhancer schedules shots within the runtime. Image input needs a *mmproj*.gguf next to the model.",
            inputs=[
                io.String.Input("hint", multiline=True, default="", tooltip="Optional theme/clue to riff on - highest priority, everything it does not specify is filled from seed-picked ingredients, so a style-only hint like '3DCG' still gets a different subject/setting/event every seed (conflicting ingredients are dropped). No need to write the duration here, it has its own input."),
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
                io.Float.Input("duration", default=10.0, min=4.0, max=300.0, step=0.5, tooltip="Target video duration in seconds - wire the same duration that feeds the sampler's frame count (single clip: 4-15; long-take chaining: the planner's total_seconds). The story is sized to fit it, and the value is stamped on the draft so the enhancer schedules shots within the runtime."),
                *sampling_inputs(),
                # Long-take chaining, wired from the H3 Chain Frame Planner. Both
                # set (>1 segments, >0 seconds) switch the draft to a numbered
                # per-segment beat sheet for the H3 Segment Beat Picker.
                io.Float.Input("segment_seconds", default=0.0, min=0.0, max=15.0, step=0.5, tooltip="Long-take chaining: wire the planner's segment_seconds. 0 = single-clip draft (default)."),
                io.Int.Input("total_segments", default=0, min=0, max=99, tooltip="Long-take chaining: wire the planner's total_segments so the draft gets exactly this many numbered beats. 0/1 = single-clip draft."),
                io.String.Input("beat_durations", default="", optional=True, tooltip="Long-take chaining: wire the planner's segment_durations (comma-separated story seconds per segment, e.g. \"8.0, 7.1, 7.1, 7.1, 0.7\") so each beat is sized to its real story time - later segments deliver less than the clip length because of the pinned overlap tail, and the last beat can be very short."),
            ],
            outputs=[io.String.Output(display_name="idea")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, hint, language, model, thinking, temperature, seed,
                keep_loaded, n_ctx, n_gpu_layers, n_cpu_moe, first_frame=None, last_frame=None,
                reference_images=None, duration=10.0,
                top_p=-1.0, top_k=-1, presence_penalty=-1.0, reasoning_effort="low",
                segment_seconds=0.0, total_segments=0, beat_durations=""):
        mode = derive_mode(first_frame, last_frame, reference_images)
        dur = f"{duration:g}"

        frames = collect_idea_frames(first_frame, last_frame, reference_images)
        user_text = build_idea_instruction(hint, frames, seed)

        image_note = f"\n- {MODE_GUIDANCE[mode]}" if frames else ""
        segment_note = build_segment_note(segment_seconds, total_segments, beat_durations)

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
                {"role": "system", "content": IDEA_SYSTEM.format(image_note=image_note, segment_note=segment_note, language=language, duration=dur)},
                {"role": "user", "content": user_content},
            ]
            log(f"idea gen: mode={mode} model={model} seed={seed} lang={language} duration={dur}s images={len(frames)} segments={total_segments}x{segment_seconds:g}s hint={hint[:100]!r}")
            log(f"user prompt: {user_text[:500]}")
            content = LocalLLM.generate(messages, max_tokens=local_options()["max_tokens"], sampling=sampling,
                                        seed=seed, on_text=make_progress_cb(cls.hidden.unique_id))
            # Stamp the runtime on the draft so the downstream enhancer can
            # schedule shot timings against the requested duration.
            header = f"总时长：{dur} 秒" if language == "中文" else f"Total runtime: {dur} seconds"
            content = f"{header}\n\n{content}"
            # 单段退化兑底：total_segments==1 时基础提示词走"无标记连续叙述"，
            # 而 BeatPicker 无条件要求分节标记（长视频 forloop 也接它）——
            # 此处确定性包一层「段1」，不依赖 LLM 是否自觉加标签。
            if total_segments == 1 and not any(
                    BEAT_LABEL.match(line.strip()) for line in content.splitlines()):
                content = f"段1：\n{content}"
            log(f"idea ({len(content)} chars):\n{content[:500]}")
            append_history({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "kind": "idea",
                "task_type": f"local/idea/{mode}",
                "model": model,
                "seed": seed,
                "duration": duration,
                "total_segments": total_segments,
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


# Beat label written by the idea generator in beat-sheet mode: "段1：...",
# "段 2 (8-16s): ...", "Segment 3 - ...". The optional bracket right after the
# number carries the beat's global time range - stripped with the label.
BEAT_LABEL = re.compile(r"^(?:段|Segment)\s*(\d+)\s*(?:\([^)]*\)|（[^）]*）)?\s*[：:.\-–]?\s*(.*)$", re.IGNORECASE)
# Hand-off line at the end of each beat (long-take mode): the exact on-screen
# state at the segment's final frame, which the next segment starts from.
END_LABEL = re.compile(r"^(?:段末|End of segment)\s*[：:]\s*(.*)$", re.IGNORECASE)

# Runtime header the idea generator stamps on the draft.
RUNTIME_HEADER = re.compile(r"总时长：\s*([0-9.]+\s*秒)|Total runtime:\s*([0-9.]+\s*seconds)", re.IGNORECASE)


class H3SegmentBeatPicker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SegmentBeatPicker",
            display_name="H3 Segment Beat Picker 长镜头节拍拾取",
            category="prompt",
            description="Pick one segment's beat from a numbered beat sheet written by the H3 Idea Generator in long-take mode (segment_seconds + total_segments wired there). Outputs the shared setup plus this segment's beat, retimed to start at 0s and marked as a seamless continuation, ready for the H3 Prompt Enhancer. Manual chaining: set clip_index by hand before each run; looped chaining: wire the loop index + 1. No LLM call - pure text splitting.",
            inputs=[
                io.String.Input("beat_sheet", multiline=True, default="", tooltip="The idea generator's draft in beat-sheet mode (labels 段1/段2/... or Segment 1/2/...). Linked input hides the box; widget kept so saved workflows round-trip."),
                io.Int.Input("clip_index", default=1, min=1, max=99, tooltip="Which segment's beat to emit (1-based, same convention as the H3 Chain Frame Planner / MotionContext)."),
                io.String.Input("segment_durations", default="", optional=True, tooltip="Optional, wire the planner's segment_durations (comma-separated story seconds per segment): the segment note states this segment's real story time so the enhancer schedules within it. Empty = omit the duration."),
            ],
            outputs=[io.String.Output(display_name="segment_prompt")],
        )

    @classmethod
    def execute(cls, beat_sheet, clip_index, segment_durations=""):
        setup_lines = []
        beats = {}
        current = None
        for line in beat_sheet.splitlines():
            m = BEAT_LABEL.match(line.strip())
            if m:
                current = int(m.group(1))
                beats.setdefault(current, [])
                if m.group(2).strip():
                    beats[current].append(m.group(2).strip())
            elif current is None:
                setup_lines.append(line)
            else:
                beats[current].append(line)
        if not beats:
            raise ValueError("no numbered beats (段1/段2/... or Segment 1/2/...) found - wire segment_seconds + total_segments on the Idea Generator and regenerate the draft.")
        if clip_index not in beats:
            raise ValueError(f"beat sheet has beats {sorted(beats)}, but clip_index={clip_index} was requested - regenerate the sheet or fix clip_index.")

        total = len(beats)
        seg = ""
        durations = [d for d in (x.strip() for x in segment_durations.split(",")) if d]
        if durations:
            try:
                secs = [float(d) for d in durations]
            except ValueError:
                raise ValueError(f"segment_durations should look like \"8.0, 7.1, 0.7\" - wire the H3 Chain Frame Planner's segment_durations output, got: {segment_durations!r}")
            if len(secs) != total:
                raise ValueError(f"beat sheet has {total} beats but segment_durations lists {len(secs)} segments - the sheet is stale, regenerate it with the current planner settings.")
            seg = f"，约 {secs[clip_index - 1]:g} 秒"
        setup = "\n".join(setup_lines).strip()
        setup = RUNTIME_HEADER.sub(
            lambda m: f"全片总时长：{m.group(1) or m.group(2)}（一镜到底，分 {total} 段链接生成）", setup, count=1)
        beat = "\n".join(beats[clip_index]).strip()

        # Hand-off anchor: the previous beat's 段末 line, so the enhancer pins
        # this segment's opening to what is actually on screen (the pinned
        # Motion-Context frames) instead of re-staging from scratch.
        prev = ""
        if clip_index > 1:
            for x in beats.get(clip_index - 1, []):
                m = END_LABEL.match(x.strip())
                if m:
                    prev = (f"上一段结束于：{m.group(1).strip()}\n"
                            "本段第一帧必须从这一状态直接继续——人物姿态、位置、运动方向与机位状态都不得跳变。\n")
                    break

        cont = "；画面与动作紧接上一段结尾无缝继续" if clip_index > 1 else "（开场段）"
        note = f"本段：一镜到底长镜头的第 {clip_index}/{total} 段{seg}，时间码从本段 0 秒记起{cont}。"
        out = f"{setup}\n\n{prev}{note}\n{beat}"
        log(f"beat pick: segment {clip_index}/{total} ({len(out)} chars)")
        return io.NodeOutput(out)
