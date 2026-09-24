from comfy_api.latest import ComfyExtension, io

from .llm_local import apply_unload_hook
from .nodes_api import DeepSeekPromptOptimizer, DeepSeekTranslator
from .nodes_cloud import (
    H3IdeaGeneratorCloud,
    H3PromptEnhancerCloud,
    H3TranslatorCloud,
    UniversalImageInterrogatorCloud,
)
from .nodes_idea import H3IdeaGeneratorLocal, H3SegmentBeatPicker
from .nodes_local import H3LLMUnload, H3PromptEnhancerLocal, H3TranslatorLocal
from .nodes_qwen21 import QwenImage21PromptEnhancerCloud
from .nodes_interrogate import UniversalImageInterrogatorLocal
from .nodes_krea2 import Krea2PromptEnhancerCloud
from .nodes_profiler import LoraProfilerLocal
from .nodes_scail2 import SCAIL2PromptGenerator, SCAIL2SegmentPlan
from .nodes_universal import UniversalPromptEnhancerLocal

apply_unload_hook()

WEB_DIRECTORY = "./web"


class H3PromptEnhancerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            H3PromptEnhancerLocal,
            H3IdeaGeneratorLocal,
            H3SegmentBeatPicker,
            H3TranslatorLocal,
            H3LLMUnload,
            UniversalPromptEnhancerLocal,
            UniversalImageInterrogatorLocal,
            LoraProfilerLocal,
            DeepSeekPromptOptimizer,
            DeepSeekTranslator,
            H3PromptEnhancerCloud,
            H3IdeaGeneratorCloud,
            H3TranslatorCloud,
            QwenImage21PromptEnhancerCloud,
            Krea2PromptEnhancerCloud,
            UniversalImageInterrogatorCloud,
            SCAIL2SegmentPlan,
            SCAIL2PromptGenerator,
        ]


async def comfy_entrypoint() -> H3PromptEnhancerExtension:
    return H3PromptEnhancerExtension()
