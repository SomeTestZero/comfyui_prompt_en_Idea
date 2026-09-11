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
| `H3 Prompt Enhancer (Cloud API)` | 云端 API 版增强器：与本地版同一套 skill、同一套图片插槽与模式推导（含 `history_entry` 跨后端回放），改用多模态云端模型生成。图片输入必须选 vision 模型，选纯文本模型接图会直接报清晰错误。 |
| `H3 Idea Generator (Cloud API) 灵感生成` | 云端 API 版灵感生成：抽卡逻辑与本地版逐位一致（同 seed 同元素池），云端模型只负责编织草稿；支持长镜头分段节拍表（`segment_seconds`/`total_segments`/`beat_durations`）。 |
| `H3 Translator (Cloud API) 审核翻译` | 云端 API 版翻译：同一套结构保留翻译规则，纯文本任务，任意模型可用。 |
| `Universal Prompt Enhancer (Local GGUF)` | 模型无关的通用版：不含 H3 视频模式推导，skill 下拉框选哪个 skill 就用哪套规则（自带 `krea2-prompt-writing`，面向 Krea 2 文生图）。支持 Autogrow 多参考图（按序对应 `<Picture 1..N>`）、`history_entry` 回放，默认生成完自动卸载模型。`prompt` 留空但接了参考图时，自动改为从图片反推意图写提示词。`lora_profile` 下拉选择 LoRA 档案后，增强器围绕档案中已验证的特征改写，不再发明与 LoRA 冲突的风格/外貌描述。 |
| `Universal Image Interrogator (Local GGUF)` | 图片反推：skill 下拉选输出风格——`img2prompt-natural` 输出自然语言段落（Krea 2/FLUX 时代），`img2prompt-tags` 输出 booru tag 列表（SD1.5/Pony/SDXL 时代）。batch 逐帧反推并拼接结果，帧间自动清 KV 防串扰。可选 `custom_instruction` 对每帧施加引导（如“不要描述水印/logo/字幕”）。需要模型配 mmproj 视觉投影。 |
| `LoRA Profiler (Local GGUF)` | LoRA 体检：喂一张"挂 LoRA + 探针词"跑出的图，视觉模型提取该 LoRA 稳定产出的特征（`probe_type=character` 提取身份长相，`style` 提取风格质感），按 LoRA 名存入 `lora_contexts.json`，供 Universal Prompt Enhancer 的 `lora_profile` 选用。配好的体检工作流见用户 workflows 目录 `LoRA体检(krea2).json`，换 LoRA 跑一次即可建档。人物 LoRA 注意两点：探针词开头必须带角色名（这批 LoRA 靠名字激活，不带名字跑出的是底模，档案就是废的）；同时把角色名填进 `trigger` 输入，增强器会强制最终提示词以触发词开头。体检时确认 LoraLoader 和 Profiler 上选的是同一个 LoRA。 |

## 模型放置

GGUF 放 `models/LLM/`，视觉投影（文件名含 `mmproj` 的 gguf）和模型放同一目录即自动配对；没有 mmproj 的模型只能纯文本，接图会报错。

默认参数按 Qwen3.8-27B（hybrid 架构：48 层线性注意力 + 16 层全注意力，KV cache 只挂在全注意力层）调好（RTX 5070 Ti 16GB + 64GB RAM 实测）：`n_gpu_layers=32` + `n_ctx=65536`（KV 约 4GiB，其中一半随 CPU 层落在内存；生成全程显存约 12.7GiB、剩 ~3.6GiB，稳定 ~7 token/s）。注意不要把层数拉满：`n_gpu_layers=48` 会把 16GB 显存顶满，Windows 驱动开始把显存页换进内存，解码断崖跌到 1 token/s 以下；ComfyUI 常驻模型多就往 24 调（余量 ~6GiB）。`n_ctx=0` 会用满 262K 原生上下文，KV 就要 16GiB，只有 MoE + `n_cpu_moe` 把权重留内存时才开得起。跑 35B-A3B MoE 的老习惯：`n_gpu_layers=-1` + `n_cpu_moe=99` + `n_ctx=0`。`thinking` 默认关闭（需要长思维链时手动开）；生成均不设输出长度上限（到 EOS 自然结束）。新节点的模型下拉默认优先选 qwen3.8 开头的模型。

`keep_loaded=False`（默认）时每次生成完自动卸载；ComfyUI 的全局释放显存操作也会连带卸载本模型。连续批量改写时开 `keep_loaded=True` 可省去重复加载。

## 采样参数（跟随思考模式，Qwen3.8 官方推荐）

本地节点的 `temperature` / `top_p` / `top_k` / `presence_penalty` 默认值都是 **-1 = auto**：自动跟随当前思考模式取 Qwen3.8 官方推荐值——

| 参数 | 思考模式（enabled） | 非思考模式（disabled） |
| --- | --- | --- |
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| presence_penalty | 0.0 | 1.5 |

把某个控件改成 ≥0 的值即对该参数单独手动接管，其余仍跟随模式。`min_p=0.0`、`repeat_penalty=1.0` 按官方值固定（旧版走 llama-cpp 默认的 0.05/1.1，与 Qwen 官方不符，2026-08-16 起修正）。`reasoning_effort`（`xhigh`/`medium`/`low`，默认 `low`，模型官方默认是 `xhigh`）控制思考深度，仅思考模式生效；模板里没有这个变量的旧模型会静默忽略。反推/翻译类任务想要旧的低温度手感，把 `temperature` 手动设 0.3 即可。

兼容性：已保存的工作流按位置存值，以上默认值变化不影响旧工作流；只有新建的节点用新默认值。解析后的实际采样参数每次生成都会写进 `h3_prompt_enhancer.log`（`sampling: {...}` 行），排查时看它。

## Skill

skill 放在包内 `skills/` 下（一个 skill 一个文件夹，含 `SKILL.md` + 可选 `references/`），启动后出现在节点的 skill 下拉框。自带 `h3-prompt-writing`（来自 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)）和 `krea2-prompt-writing`（依据 [Krea 2 官方 prompting 指南](https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md) 编写，references 含官方示例与 expansion 规则）。

路由规则按 skill 官方约定：T2VA/I2VA/FL2VA/L2VA 读 `references/base-en.txt`，Ref2VA 读 `references/ref-en.txt`；其他布局（如 krea2）的 skill 读取 `references/` 下全部 `.txt`/`.md` 文件。

## 配置

DeepSeek API key 写在 `config.json`（见 `config.example.json`）或节点的 `api_key` 输入。运行历史存 `prompt_history.jsonl`，日志 `h3_prompt_enhancer.log`。

## 云端 API 节点（Cloud API 三件套）

三个云端节点（增强器/灵感生成/翻译）共用一个 `model` 下拉，选项为 `provider/model` 形式的 OpenAI 兼容端点组合：

| provider | 端点 | 内置模型 | API key 环境变量 |
| --- | --- | --- | --- |
| `volcengine` | 方舟按量端点 `/api/v3` | doubao-seed-2-1-pro/turbo-260628、doubao-seed-evolving、doubao-seed-2-0-pro-260215（均支持图片） | `VOLCENGINE_API_KEY` |
| `volcengine-plan` | Agent Plan 套餐 `/api/plan/v3` | glm-5.3-flash（支持图片）、glm-5.3 | `VOLCENGINE_AGENT_PLAN_API_KEY` |
| `volcengine-coding` | Coding Plan 套餐 `/api/coding/v3` | glm-5.3-flash（支持图片）、glm-5.3 | `VOLCENGINE_CODING_PLAN_API_KEY` |
| `deepseek` | DeepSeek API | deepseek-v4-flash-vision-exp（支持图片）、deepseek-v4-flash、deepseek-v4-pro | `DEEPSEEK_API_KEY` |

模型清单手动维护（与 pi-volcengine-plans 插件同思路，无远程拉取、离线可用）；加模型往 `nodes_cloud.py` 的 `MODELS` 里复制一行，或在 `config.json` 的 `cloud.<provider>.models` 追加（自定义 provider 同理，需给 `base_url`）。key 来源优先级：节点 `api_key` 输入 > `config.json` `cloud.<provider>.api_key`（deepseek 还会回退旧顶层 `api_key`）> 对应环境变量。

行为差异说明：

- `thinking` 下拉对应 OpenAI 兼容 `thinking.type` 参数（豆包 Seed/DeepSeek 支持）；GLM-5.3 系思考常开、忽略该开关，深度由 `reasoning_effort` 控制（low/medium/high 映射 low/high/max）。
- `reasoning_effort`：豆包直发，`auto` = 不发该参数（服务端默认）；DeepSeek 忽略。
- `seed` 同时驱动本地抽卡与 API 请求；云端服务只能尽力复现，抽卡元素本身是完全可复现的。
- 云端节点的输出/历史与本地节点写同一份 `prompt_history.jsonl`，`history_entry` 可跨后端回放（记录带 `cloud/`、`local/` 前缀区分）。
- 火山方舟多模态传图用 OpenAI 兼容 `image_url`（base64 data URL，长边 768 缩放后 JPEG），与本地 mmproj 同一套图片预处理。
