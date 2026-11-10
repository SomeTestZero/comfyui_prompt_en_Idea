# H3 Prompt Enhancer

ComfyUI 自定义节点包：用本地多模态 GGUF 模型（llama.cpp）、DeepSeek API 或任意 OpenAI 兼容云端（火山方舟 Agent/Coding Plan、DeepSeek V4 等），按 MiniMax H3 官方 skill 把粗略想法改写成结构化 H3 视频提示词。

## 节点

| 节点 | 说明 |
| --- | --- |
| `H3 Prompt Enhancer (Local GGUF)` | 本地 GGUF 后端。图片输入按语义分三类插槽：`first_frame`/`last_frame`（首尾帧）和 `reference_images`（Autogrow 多参考图，插槽 `ref_image_0..8` 与 MiniMax H3 Reference to Video 节点同名同序——同一张图接两边同号插槽即可，按序对应 `<Picture 1..N>`）。`task_type=Auto` 时按接线确定模式：有参考图→Ref2VA、首+尾→FL2VA、仅首→I2VA、仅尾→L2VA、无图→T2VA（确定性推导，无额外调用）。参考图和帧同时接时帧作为锚点 Picture 追加。默认生成完自动卸载模型。支持 `history_entry` 下拉回放历史记录（与 DeepSeek 版共享 `prompt_history.jsonl`），回放时不加载模型。 |
| `Unload Local LLM (H3)` | 手动卸载本地模型（透传节点），配合 `keep_loaded=True` 使用。 |
| `H3 Idea Generator (Local GGUF) 灵感生成` | 没灵感时用：本地 GGUF 编一个故事草稿（一句铺垫 + 分镜节拍，含台词；分镜数量随故事而定，长镜头或闪切皆可，可直接当提示词素材），输出接 `H3 Prompt Enhancer` 的 `prompt`。图片输入与增强器完全镜像（`first_frame`/`last_frame`/`ref_image_0..8`，同一批图两边同名插槽各接一份）：接首尾帧→围绕帧画面编故事（首+尾帧则故事从首帧发展到尾帧）；接参考图→以图中人物/主体为主角编故事；不接图→按 seed 从内置元素池（类型/主角/场景/核心事件/基调；微观世界与古风类型、地点绑定的主角会走相容子池，上亿种组合但不会自相矛盾；核心事件混合安静日常与少量奇想事件，不强制转折）随机抽取编织，每个 seed 一个梗且可复现，避免模型自己"随机"时翻来覆去那几个套路。接图后随机元素只抽"意外元素+基调"（主角场景来自画面），换 seed 就是同一角色的另一个故事。`hint` 填了则以 hint 为最高优先级（可与图片叠加，时长不用写在这里）：hint 覆盖到的方面以 hint 为准、与之冲突的随机元素直接弃用，hint 没提的方面由 seed 抽的随机元素补足--所以只写个风格词（如“3DCG”）也能每个 seed 换题材，不会坍缩成同一个套路；hint 里的明确要求如风格、镜头策略会在草稿开头行原样复述并附英文术语，确保传到增强器不丢失。`duration` 输入目标时长（4–15 秒，默认 10，与采样器的时长接同一个源）：故事按它控制节奏，且草稿开头会标注总时长，供增强器在时长内排布分镜时间点。接图需模型配 mmproj。流水线：灵感生成 → 增强 → 审核翻译。 |
| `H3 Translator (Local GGUF) 审核翻译` | 本地 GGUF 翻译，用于快速审核增强后的提示词。模型参数默认值与增强节点一致：增强节点开 `keep_loaded=True`、串到翻译节点，全程只加载一次模型，翻译完自动卸载。 |
| `H3 Segment Beat Picker 长镜头节拍拾取` | 从灵感生成的长镜头分段草稿（接了 `segment_seconds` + `total_segments` 时生成的「段1/段2/...」编号节拍表）中拾取一段：输出共享铺垫 + 本段节拍，时间码重置为本段 0 秒并标注无缝续接，直接接 `H3 Prompt Enhancer` 的 `prompt`。手动续链时逐次改 `clip_index`，循环续链时接循环索引 +1；接上规划器的 `segment_durations` 还会在段注释里写明本段真实故事时长。纯文本拆分，不调模型。 |
| `DeepSeek Prompt Optimizer` | DeepSeek API 后端（原 deepseek_prompt_optimizer，不支持图片输入）。 |
| `DeepSeek Translator (审核翻译)` | DeepSeek API 翻译。 |
| `H3 Prompt Enhancer (Cloud API)` | 云端 API 版增强器：与本地版同一套 skill、同一套图片插槽与模式推导（含 `history_entry` 跨后端回放），改用多模态云端模型生成。图片输入必须选 vision 模型，选纯文本模型接图会直接报清晰错误。`lora_preset` 选动作/风格 LoRA 预设（中文下拉：不注入 / 武打·三件套（拳脚·稳）/ （兵器·稳）/ （拳脚·强打）/ （兵器·强打）/ 武打·单开 Combat V2 / 武打·单开 Motion_Repair V2；数据在 `lora_presets.json`，键名即下拉文字，**不进 skills/**，官方 skill 保持原样；它只管写词，**不加载 LoRA**）：预设的激活词会被要求逐字写进 `detailed_description` 末尾的 cue 行，模型若漏掉则由代码兜底补上（插在 `overall_soundscape` 之前），同时给改写加一条该 LoRA 的动作/构图约束。改完 `lora_presets.json` 需刷新页面让下拉框重读。 |
| `H3 Idea Generator (Cloud API) 灵感生成` | 云端 API 版灵感生成：抽卡逻辑与本地版逐位一致（同 seed 同元素池），云端模型只负责编织草稿；支持长镜头分段节拍表（`segment_seconds`/`total_segments`/`beat_durations`）。 |
| `H3 Translator (Cloud API) 审核翻译` | 云端 API 版翻译：同一套结构保留翻译规则，纯文本任务，任意模型可用。 |
| `Universal Prompt Enhancer (Local GGUF)` | 模型无关的通用版：不含 H3 视频模式推导，skill 下拉框选哪个 skill 就用哪套规则（自带 `krea2-prompt-writing`，面向 Krea 2 文生图）。支持 Autogrow 多参考图（按序对应 `<Picture 1..N>`）、`history_entry` 回放，默认生成完自动卸载模型。`prompt` 留空但接了参考图时，自动改为从图片反推意图写提示词。`lora_profile` 下拉选择 LoRA 档案后，增强器围绕档案中已验证的特征改写，不再发明与 LoRA 冲突的风格/外貌描述。 |
| `Universal Image Interrogator (Local GGUF)` | 图片反推：skill 下拉选输出风格——`img2prompt-natural` 输出自然语言段落（Krea 2/FLUX 时代），`img2prompt-tags` 输出 booru tag 列表（SD1.5/Pony/SDXL 时代）。batch 逐帧反推并拼接结果，帧间自动清 KV 防串扰。可选 `custom_instruction` 对每帧施加引导（如“不要描述水印/logo/字幕”）。需要模型配 mmproj 视觉投影。 |
| `Universal Image Interrogator (Cloud API)` | 云端 API 版图片反推：与本地版同一套 skill（`img2prompt-natural`/`img2prompt-tags`）、同一套 `custom_instruction` 引导、同一套逐帧反推拼接（每帧一次调用，结果间插空行分隔符，与本地版完全一致），改用多模态云端模型生成。图片输入必须选 vision 模型，选纯文本模型接图会直接报清晰错误。`history_entry` 可回放本地/云端的反推记录（不调 API）。 |
| `Qwen Image 2.1 Prompt Enhancer (Cloud API)` | Qwen-Image 2.1 专用增强器（任务类型：文生图/编辑/人物卡三模式，下拉全汉字——内部模式串不变，旧工作流英文值宽容放行）：把粗略需求改写成 `TextEncodeQwenImage21` 可直接用的提示词，依据官方 PE-T2I/PE-I2I 提示词改写模型的 system prompt（skill `qwen-image21-prompt-writing`，官方原文存档 + 提炼规则 + 人物卡组合指南，按模式路由只加载对应一份）。图片插槽 `image_1..N` 与 `TextEncodeQwenImage21` 的 `images.image_N` 同名同序（同一张图两边同号插槽各接一份），改写后按官方规则以 `<image1>`/`<image2>` 编址（单图自然指代不打标签）；`prompt` 留空时从接图反推编辑意图。`task_type=Auto` 时接了图→Edit、没图→T2I；`task_type=CharacterSheet` = 人设图/角色参考表专用（正/侧/背三视图 + 面部特写，默认四格横排、纯色背景、脚下无站位盘、整版不带文字——用户点名要标题/标签才按逐字规则加；硬规则=角色外观全文只写一次、面板只变机位防多视图漂移；无图从零构建，接图则身份锚定 `<image1>` 只写版式），路由 `charactersheet-en.txt`。输入 `background=Transparent` 切 RGBA 透明底（官方透明措辞整段包裹、不写任何背景），`Auto` = 普通背景（人物卡模式自动选与人物主色反差的纯色底：浅人物深底/深人物浅底，用户点名的颜色优先）。输入 `style_preset` 选预设风格块（内置写实游戏CG/电影实拍/日漫赛璐璐/漫画墨线/水彩/油画/扁平矢量/仙侠水墨八档，逐字钉入不许改写）或 `custom` 后在 `style_profile` 自由输入风格块：一套图共用同一段即画风统一；风格也可用参考图锚定（“照 <imageN> 的风格画”只取其画风不搬主体）。提示词内不写分辨率/宽高比（由工作流的分辨率节点负责）；渲染文字用双引号逐字锁定。云端模型/密钥/`history_entry` 口径与 H3 Cloud 版一致。 |
| `Krea 2 Prompt Enhancer (Cloud API)` | Krea 2 专用增强器（任务类型：文生图/人物卡两模式，下拉全汉字——内部模式串不变，旧工作流英文值宽容放行）：把粗略需求改写成 Krea2 CLIPTextEncode 可直接用的英文段落，依据官方 Krea 2 提示词指南 + Krea 自家扩写 system prompt（skill `krea2-prompt-writing`）。`task_type=人物卡` = 人设图/角色参考表专用（正/侧/背三视图 + 面部特写；版式按用户理想样张：三全身格等宽占左 2/3 + 特写格更宽占右 1/3、水泥墙影棚背景、每全身格脚下一个细圆环标记、特写裁切到锁骨/上胸、整版不带文字——用户点名要标题/标签才按逐字规则加；硬规则=角色外观全文只写一次、每格以横向位置词开头绑定机位+每格写明可见面防多视图漂移、段尾收一句“共四格左→右：正/侧/背/特写”防 Krea 2 复制面板（实测会多出一格重复背面）；**外观不发明**：只写用户明说的话/参考图读出的/LORA 档案实测特征，光给角色名时外观整段留空交给工作流里挂的角色 LoRA（名字/触发词逐字保留不翻译，防 LoRA 点不着）），路由 `charactersheet-en.txt`（250–350 词，比 Qwen 版短：Krea 2 自然语言段落不留长篇说明文）。接参考图时外观从图上读出写进唯一外观清单（Krea 2 生成纯文本：提示词里不提参考图/不打图签）；无图从零构建。云端多模态模型口径（provider/model/thinking/effort/history_entry）与 Qwen 2.1 版一致；参考图只作描述依据（接图且 prompt 留空时反推意图）；`style_preset`/`style_profile` 与 Qwen 版同套风格预设钉字块；`lora_profile` 接 LoRA Profiler 档案（注入实测特征 + 强制触发词开头，配 `Kera2\角色K2-*` LoRA 库使用）。 |
| `MOSS Prompt Enhancer (Local GGUF)` / `(Cloud API)` | MOSS-Image 专用增强器（任务类型：文生图/版式布局），依据 OpenMOSS 官方 `moss-image-prompt-enhancer` skill（skill `moss-image-prompt-writing`，官方 SKILL.md 与 layout.md 原文存档 + 提炼的 caption/layout 指南，按模式路由只加载对应一份）。文生图模式写「成品图描述」英文段落（80–200 词，成片视角、只写可见内容、画面内文字原语言逐字引号锁定、不写画质词/负向词）；版式布局模式让模型输出 `abs`/`xyxy` 像素 bbox 草案，节点按官方编译口径（floor 左/上、ceil 右/下再减画布中心）转成模型要的 `abs_centralized`/`yxyx` 单行 JSON，并校验条目/标签/文字框/越界与 width/height 是否与画布一致——不再需要官方那个 Python 编译脚本。`width`/`height` 同时是版式坐标画布，工作流里与 `EmptyFlux2LatentImage` 同源即可保证提示词与采样分辨率一致。本地/云端两版共用同一套 skill、编译与 `history_entry`（跨后端回放），参数口径同 H3/Krea 版。 |
| `H3 Swap Prompt (Cloud API) 人物替换提示词` | 人物替换专用：`instruction` 一句话说清"换谁+换成谁"（名字/触发词逐字不翻译），云端多模态模型看**源视频帧批**（`source_frames` 接 VHS_LoadVideo 的 IMAGE 输出，内部均匀采样 6 帧带时间标注）指认被换人、写动作/镜头，再看角色参考图写身份，输出六段式替换提示词直连 MiniMax H3 Reference to Video 的 prompt。`ref_image_N` 与 R2V 同名同号；`duration` 接采样同款，用于 [Shot N] 时间戳。写作规则在 `h3-prompt-writing` 的 `characterswap-en.txt` 键控指南（仅本任务加载；铁律=外观只写可见+用户明说、`<Subject 2>` 衣着/位置唯一指认、防照抄条款、时序一致性条款、参考图不上屏）。输出自带 lint（六段齐全/标签越界）不合格打 warning 日志。`history_entry` 回放口径与 H3 Cloud 版一致。 |
| `H3 Swap Plan 全片人物与分段提示词` / `H3 Swap Segment Prompt 当前段换人提示词` | 分段换人入口：全片一次多模态调用，默认 **每 2 秒一帧**并补各段首尾、MC 头锚交界和交付末帧（长边 768），按首次出场建立稳定 P1/P2 编号，记录每个源窗口实际在场的人和局部事件；光照、人数、服装状态随源变化。`sample_length/context_length/delivered_frames` 分别接分段器的 `length/overlap/total_delivered`；取词节点接当前段号、`skip_first_frames/length/head_trim/deliver`。低清/高清从同一计划取同段提示词，无追加 API 调用。支持 `ref_image_0 替换人物1，ref_image_1 替换人物2`；同一参考图必须接到 AI 与两个 Ref2V 同号插槽。参考图按连接顺序对应 Picture，缺席人物不进入本段描述，不重编号；源片尾事件按源帧定位到对应段，不复制到每段。歧义映射、越界事件或过期窗口先报错，采样不执行；稀疏观察仍可能漏掉短暂出入场，可调 0.5–1 秒并先看人物/分段预览。回放校验素材指纹、参考图、观察帧及窗口；旧单片提示词历史不能当分段计划。原 `H3 Swap Prompt` 仍用于单片，禁止将它的整片时间轴原样复用于每个分段。 |
| `LoRA Profiler (Local GGUF)` | LoRA 体检：喂一张"挂 LoRA + 探针词"跑出的图，视觉模型提取该 LoRA 稳定产出的特征（`probe_type=character` 提取身份长相，`style` 提取风格质感），按 LoRA 名存入 `lora_contexts.json`，供 Universal Prompt Enhancer 的 `lora_profile` 选用。配好的体检工作流见用户 workflows 目录 `LoRA体检(krea2).json`，换 LoRA 跑一次即可建档。人物 LoRA 注意两点：探针词开头必须带角色名（这批 LoRA 靠名字激活，不带名字跑出的是底模，档案就是废的）；同时把角色名填进 `trigger` 输入，增强器会强制最终提示词以触发词开头。体检时确认 LoraLoader 和 Profiler 上选的是同一个 LoRA。 |

## 模型放置

GGUF 放 `models/LLM/`，视觉投影（文件名含 `mmproj` 的 gguf）和模型放同一目录即自动配对；没有 mmproj 的模型只能纯文本，接图会报错。

默认参数按 Qwen3.8-27B（hybrid 架构：48 层线性注意力 + 16 层全注意力，KV cache 只挂在全注意力层）调好（RTX 5070 Ti 16GB + 64GB RAM 实测）：`n_gpu_layers=32` + `n_ctx=65536`（KV 约 4GiB，其中一半随 CPU 层落在内存；生成全程显存约 12.7GiB、剩 ~3.6GiB，稳定 ~7 token/s）。注意不要把层数拉满：`n_gpu_layers=48` 会把 16GB 显存顶满，Windows 驱动开始把显存页换进内存，解码断崖跌到 1 token/s 以下；ComfyUI 常驻模型多就往 24 调（余量 ~6GiB）。`n_ctx=0` 会用满 262K 原生上下文，KV 就要 16GiB，只有 MoE + `n_cpu_moe` 把权重留内存时才开得起。跑 35B-A3B MoE 的老习惯：`n_gpu_layers=-1` + `n_cpu_moe=99` + `n_ctx=0`。`thinking` 默认关闭（需要长思维链时手动开）；生成均不设输出长度上限（到 EOS 自然结束）。新节点的模型下拉默认优先选 qwen3.8 开头的模型。

送进视觉编码的图按长边缩放：参考图/探针图走 `IMAGE_LONG_EDGE=768`（多图省钱），**反推节点单独用 `INTERROGATE_LONG_EDGE=1344`**（两者都可在 `config.json` 的 `local` 段覆盖）。视觉 token 数是 `(宽/32)*(高/32)`，16:9 图 768 长边只有 ~336 token、1344 长边 ~1008 token，而 llama.cpp 加载时会提示 Qwen-VL 需要 ≥1024 个图像 token 才不掉精度——反推既只有一张图、prefill 又只差 ~1 秒，没理由省这点。

`keep_loaded=False`（默认）时每次生成完自动卸载；ComfyUI 的全局释放显存操作也会连带卸载本模型。连续批量改写时开 `keep_loaded=True` 可省去重复加载。

## 采样参数（跟随思考模式，Qwen3.8 官方推荐）

本地节点的 `temperature` / `top_p` / `top_k` / `presence_penalty` 默认值都是 **-1 = auto**：自动跟随当前思考模式取 Qwen3.8 官方推荐值——

| 参数 | 思考模式（enabled） | 非思考模式（disabled） |
| --- | --- | --- |
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| presence_penalty | 0.0 | 1.5 |

把某个控件改成 ≥0 的值即对该参数单独手动接管，其余仍跟随模式。`min_p=0.0`、`repeat_penalty=1.0` 按官方值固定（旧版走 llama-cpp 默认的 0.05/1.1，与 Qwen 官方不符，2026-08-16 起修正）；惩罚类参数实际只看最近 `penalty_last_n` 个 token（llama-cpp 默认 64，可在 config.json 调，见下节）。`reasoning_effort`（`xhigh`/`medium`/`low`，默认 `low`，模型官方默认是 `xhigh`）本该控制思考深度，**但带 mmproj 时实际无效**：llama-cpp-python 的 MTMD handler 用它自己的内置 Qwen 模板渲染（`_resolve_chat_format` 里 `self.chat_format` 已被内置模板占位，GGUF 内嵌模板中定义 `reasoning_effort` 的那段根本没参与），内置模板里没有这个变量——实测 `low` 与 `xhigh` 两次生成的 token 数（544/316/227）与输出完全一致。要让它生效得把 GGUF 的 `tokenizer.chat_template` 读出来经 `chat_template_override` 交给 handler，但那会同时换掉多图编号（`Picture N`）与系统消息合并的渲染方式，需重新验证，暂未做。反推/翻译类任务想要旧的低温度手感，把 `temperature` 手动设 0.3 即可。

`max_tokens` 默认 `-1`（不设上限，到 EOS 自然结束）：思考内容和答案共用这一个生成预算、推理先花，所以同一段输入开思考后耗的 token 成倍上涨（实测反推一帧：思考关 196 token，思考开 544＝316 推理＋227 答案）。任何截断（不论停在推理段还是答案段）节点都直接报错要求调大，而不是把半截内容当结果返回；批量反推想给每帧留个安全阀，就把 `max_tokens` 设成 `-1` 以外的值，4096 已经比实测用量宽一个量级。

兼容性：已保存的工作流按位置存值，以上默认值变化不影响旧工作流；只有新建的节点用新默认值。解析后的实际采样参数每次生成都会写进 `h3_prompt_enhancer.log`（`sampling: {...}` 行），排查时看它。

## Skill

skill 放在包内 `skills/` 下（一个 skill 一个文件夹，含 `SKILL.md` + 可选 `references/`），启动后出现在节点的 skill 下拉框。自带 `h3-prompt-writing`（来自 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)）、`krea2-prompt-writing`（依据 [Krea 2 官方 prompting 指南](https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md) 编写，references 含官方示例与 expansion 规则）和 `qwen-image21-prompt-writing`（面向 Qwen-Image-2.1 文生图/编辑，references 为官方 PE-T2I/PE-I2I 提示词改写模型 system prompt 的原文存档——即 Qwen-Image-2.1 README 推荐的官方“怎么写提示词”口径）以及 `moss-image-prompt-writing`（来自 [OpenMOSS/MOSS-Image 官方 skill](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer)，references 存档官方 SKILL.md 与 layout.md，另提炼 caption/layout 两份键控指南：T2I→`t2i-en.txt`、Layout→`layout-en.txt`）。

路由规则按 skill 官方约定：T2VA/I2VA/FL2VA/L2VA 读 `references/base-en.txt`，Ref2VA 读 `references/ref-en.txt`；否则模式名小写匹配的 `references/{mode}-en.txt` 存在时只读它（qwen-image21-prompt-writing：T2I→`t2i-en.txt`、Edit→`edit-en.txt`、CharacterSheet→`charactersheet-en.txt`；krea2-prompt-writing：CharacterSheet→`charactersheet-en.txt`）；都没有则读取 `references/` 下的一般性 `.txt`/`.md` 文件（模式键控指南不掺入，防止串模式；skill 里全是键控指南时才全量读）。

## 配置

DeepSeek API key 写在 `config.json`（见 `config.example.json`）或节点的 `api_key` 输入。运行历史存 `prompt_history.jsonl`，日志 `h3_prompt_enhancer.log`（两者超过 8MB 自动裁尾）。

### 本地后端隐藏参数（`config.json` 的 `local` 段）

本地节点的控件只有 model / thinking / temperature / seed / keep_loaded / n_ctx / n_gpu_layers / n_cpu_moe + 采样组，其余后端参数放在 `config.json`——**不要往已发布节点插新控件**：工作流按位置存值（且尾部还有 `history_entry` 这类可接线控件），插在中间会让旧工作流的值整体错位。`local` 段每次运行重新读取（改 `kv_type`/`image_*_tokens` 会触发模型重新加载），写错过键名会直接报错并列出可用键：

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `image_long_edge` | 768 | 参考图 / LoRA 探针图送视觉编码的长边 |
| `interrogate_long_edge` | 1344 | 反推单帧的长边（见上文视觉 token 推算） |
| `jpeg_quality` | 85 | 编码 JPEG 质量，只影响细节保留与 base64 体积 |
| `kv_type` | `""`（=f16） | KV cache 精度，如 `q8_0`：n_ctx=65536 时 KV 约从 4GiB 降到 2GiB，省下的显存可以把 `n_gpu_layers` 往上提；`q4_0` 更省但可能掉质量 |
| `image_min_tokens` | -1 | mtmd 视觉 token 下限（-1=按模型元数据）。就是 llama.cpp 那条“Qwen-VL 需要 ≥1024 图像 token”警告的开关，设 1024 会把小图主动放大补足 |
| `image_max_tokens` | -1 | mtmd 视觉 token 上限：给超大输入封顶，免得 2K/4K 图把 prefill 吃满 |
| `max_tokens` | -1 | 给**没有 max_tokens 控件的节点**（H3 增强器/翻译/灵感生成）设一个统一生成上限，-1=到 EOS；有控件的节点仍以控件为准，不读这个键 |
| `penalty_last_n` | 64 | repeat/present 惩罚能看到的 token 窗口（llama.cpp 默认 64，`-1`=整个上下文）。Qwen 官方 `presence_penalty=1.5` 的口径是全序列，嫌输出复读就调大（如 512） |

云端节点不受这些影响（图片编码固定 768 长边，API 按图像 token 计费）。

## 云端 API 节点（Cloud API 五件套）

九个云端节点（H3 增强器/灵感生成/翻译/图片反推、Qwen-Image 2.1、Krea 2、MOSS、H3 Swap、SCAIL-2 生成器）共用一个 `model` 下拉，选项为 `provider/model` 形式的 OpenAI 兼容端点组合：

| provider | 端点 | 内置模型 | 鉴权 |
| --- | --- | --- | --- |
| **`workbuddy`** ★默认 | `copilot.tencent.com/v2` | **deepseek-v4.1-flash（默认，1M ctx/128K out/视觉/可关思考）**、deepseek-v4-pro、glm-5.3-flash、kimi-k3-1、minimax-m3.1-flash、space-bunny（全部支持图片） | **OAuth，无需 api_key** |
| `volcengine` | 方舟按量端点 `/api/v3` | doubao-seed-2-1-pro/turbo-260628、doubao-seed-evolving、doubao-seed-2-0-pro-260215（均支持图片） | `VOLCENGINE_API_KEY` |
| `volcengine-plan` | Agent Plan 套餐 `/api/plan/v3` | 2026-09-13 实测：vision 可用 = glm-5.3-flash、doubao-seed-2-1-turbo / 2.0-pro / 2.0-lite、doubao-seed-evolving、kimi-k3 / k2.7-code；纯文本 = glm-5.3 / 5.2 / 5.3v；**deepseek-v4 全系与 minimax-m3 会静默忽略图片，已标为不可接图** | `VOLCENGINE_AGENT_PLAN_API_KEY` |
| `volcengine-coding` | Coding Plan 套餐 `/api/coding/v3` | glm-5.3-flash（支持图片）、glm-5.3 | `VOLCENGINE_CODING_PLAN_API_KEY` |
| `deepseek` | DeepSeek API | deepseek-v4-flash-vision-exp（支持图片）、deepseek-v4-flash、deepseek-v4-pro | `DEEPSEEK_API_KEY` |

**`workbuddy` 的鉴权**：WorkBuddy/CodeBuddy 订阅额度，token 是 OAuth 访问令牌（约 28 天过期），
由 `workbuddy.py` 直接读 pi CLI 的登录态 `~/.pi/agent/auth.json`；临期或遇 401 时自动调
`/v2/plugin/auth/token/refresh` 续期，并把轮换后的 access+refresh 一起写回该文件（pi 和 ComfyUI 共用一份登录，
不会互相踢掉）。出站还带 `x-agent-*` / `x-product` 等归因头，否则用量不记到 WorkBuddy 使用端。
换机器或过期后只要在 pi 里 `/login workbuddy` 一次即可，节点侧不用改配置。
**注意**：这是非官方客户端用法，订阅协议只覆盖 WorkBuddy/CodeBuddy 自家产品，账号风险自担；
官方合规路径是 Coding Plan / Token Plan 的 API Key（见上面三个 volcengine 档）。

模型清单手动维护（与 pi-volcengine-plans 插件同思路，无远程拉取、离线可用）；加模型往 `nodes_cloud.py` 的 `MODELS` 里复制一行，或在 `config.json` 的 `cloud.<provider>.models` 追加（自定义 provider 同理，需给 `base_url`）。key 来源优先级：节点 `api_key` 输入 > `config.json` `cloud.<provider>.api_key`（deepseek 还会回退旧顶层 `api_key`）> 对应环境变量。

行为差异说明：

- `thinking` 下拉对应 OpenAI 兼容 `thinking.type` 参数（豆包 Seed/DeepSeek 支持）；GLM-5.3 系思考常开、忽略该开关，深度由 `reasoning_effort` 控制（low/medium/high 映射 low/high/max）。
- `reasoning_effort`：豆包直发，`auto` = 不发该参数（服务端默认）；DeepSeek 忽略。
- `seed` 同时驱动本地抽卡与 API 请求；云端服务只能尽力复现，抽卡元素本身是完全可复现的。
- 云端节点的输出/历史与本地节点写同一份 `prompt_history.jsonl`，`history_entry` 可跨后端回放（记录带 `cloud/`、`local/` 前缀区分）；下拉只列同类记录（增强/灵感/反推按 `kind` 各自过滤，反推列表不会混进增强结果）。
- 火山方舟多模态传图用 OpenAI 兼容 `image_url`（base64 data URL，长边 768 缩放后 JPEG），与本地 mmproj 同一套图片预处理。
- 2026-09-14 实测：`UniversalImageInterrogatorCloud` 在 `volcengine-plan/glm-5.3-flash` 上通过红/绿定位测试（vision 真实生效），`img2prompt-natural`/`img2prompt-tags` 两个 skill 的输出格式均正确，`custom_instruction` 引导与 `history_entry` 回放均生效。
