"""节点菜单分类：所有自写节点统一挂在 `yyp/` 下，显示名前缀 `[yyp] `。

分类树（与 comfyui-yyp-nodes/categories.py 保持同一套口径，两包独立不互相 import）：

    yyp/prompt/enhance     提示词增强器（H3/Qwen/Krea2/MOSS/Universal/Swap）
    yyp/prompt/translate   审核翻译
    yyp/prompt/idea        灵感生成 / 长镜头节拍 / SCAIL-2 自动分段
    yyp/prompt/llm         看图反推、LoRA 画像、LLM 卸载
"""

ROOT = "yyp"

PROMPT_ENHANCE = ROOT + "/prompt/enhance"
PROMPT_TRANSLATE = ROOT + "/prompt/translate"
PROMPT_IDEA = ROOT + "/prompt/idea"
PROMPT_LLM = ROOT + "/prompt/llm"


def name(text: str) -> str:
    """给显示名加 yyp 前缀（已加过的原样返回）。"""
    return text if text.startswith("[yyp] ") else "[yyp] " + text
