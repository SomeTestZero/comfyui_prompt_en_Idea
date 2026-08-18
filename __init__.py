from comfy_api.latest import ComfyExtension, io

from .llm_local import apply_unload_hook
from .nodes_api import DeepSeekPromptOptimizer, DeepSeekTranslator
from .nodes_idea import H3IdeaGeneratorLocal
from .nodes_local import H3LLMUnload, H3PromptEnhancerLocal, H3TranslatorLocal
from .nodes_interrogate import UniversalImageInterrogatorLocal
from .nodes_profiler import LoraProfilerLocal
from .nodes_universal import UniversalPromptEnhancerLocal

apply_unload_hook()

WEB_DIRECTORY = "./web"


class H3PromptEnhancerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            H3PromptEnhancerLocal,
            H3IdeaGeneratorLocal,
            H3TranslatorLocal,
            H3LLMUnload,
            UniversalPromptEnhancerLocal,
            UniversalImageInterrogatorLocal,
            LoraProfilerLocal,
            DeepSeekPromptOptimizer,
            DeepSeekTranslator,
        ]


async def comfy_entrypoint() -> H3PromptEnhancerExtension:
    return H3PromptEnhancerExtension()
