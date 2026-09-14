import base64
import gc
import io
import os
import time

# torch/MKL (libiomp5md) and llama.cpp's bundled libomp140 would otherwise abort
# the process with OMP Error #15 the first time the vision encoder runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import folder_paths
import comfy.model_management as mm

from .common import load_config, log

IMAGE_LONG_EDGE = 768  # reference images are resized to this before encoding
# A lone interrogation frame gets more pixels: llama.cpp warns Qwen-VL wants >=1024
# image tokens, and the vision encoder burns (w/32)*(h/32) of them - 768 long edge is
# only ~336 for a 16:9 frame, 1344 is ~1008, for the same ~1s of prefill.
INTERROGATE_LONG_EDGE = 1344

# Hidden knobs of the GGUF backend, overridable per install from config.json's "local"
# section (see README): image quality/size and the two load-time options that matter on
# a 16GB card. Read on every use, so a saved edit applies to the next run.
LOCAL_OPTIONS = {
    "image_long_edge": IMAGE_LONG_EDGE,
    "interrogate_long_edge": INTERROGATE_LONG_EDGE,
    "jpeg_quality": 85,
    "kv_type": "",           # ggml KV cache type name, e.g. q8_0 (halves the ~4GiB KV); "" = f16
    "image_min_tokens": -1,  # mtmd vision token floor/ceiling, -1 = whatever the model metadata says
    "image_max_tokens": -1,
    "max_tokens": -1,        # completion cap for the nodes that have no max_tokens widget; -1 = to EOS
    "penalty_last_n": 64,    # tokens the repeat/present penalties see (llama-cpp default; -1 = whole context)
}


def local_options():
    overrides = load_config().get("local") or {}
    unknown = sorted(set(overrides) - set(LOCAL_OPTIONS))
    if unknown:
        raise ValueError(f"unknown key(s) under \"local\" in config.json: {', '.join(unknown)}; supported: {', '.join(LOCAL_OPTIONS)}")
    for key, value in overrides.items():
        if not isinstance(value, type(LOCAL_OPTIONS[key])):
            raise ValueError(f'config.json "local".{key} must be {type(LOCAL_OPTIONS[key]).__name__}, got {value!r}')
    return {**LOCAL_OPTIONS, **overrides}


def resolve_kv_type(name):
    """llama.cpp wants a ggml type id for type_k/type_v; accept the usual names."""
    from llama_cpp._ggml import GGMLType

    key = f"GGML_TYPE_{name.strip().upper()}"
    if not hasattr(GGMLType, key):
        raise ValueError(f'config.json "local".kv_type must be a ggml type name such as q8_0 or f16, got {name!r}')
    return int(getattr(GGMLType, key))


def register_llm_folder():
    if "LLM" not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))


def list_gguf_models():
    register_llm_folder()
    try:
        names = folder_paths.get_filename_list("LLM")
    except KeyError:
        return []
    models = [n for n in names if n.lower().endswith(".gguf") and "mmproj" not in os.path.basename(n).lower()]
    return sorted(models)


def find_mmproj(model_rel_path):
    """mmproj gguf living next to the model file, or None for text-only."""
    register_llm_folder()
    model_path = folder_paths.get_full_path("LLM", model_rel_path)
    if model_path is None:
        raise FileNotFoundError(f"Model not found under models/LLM: {model_rel_path}")
    model_dir = os.path.dirname(model_path)
    for name in sorted(os.listdir(model_dir)):
        if "mmproj" in name.lower() and name.lower().endswith(".gguf"):
            return os.path.join(model_dir, name)
    return None


def tensor_to_base64_jpeg(image, max_long_edge=None):
    """ComfyUI IMAGE tensor ([H,W,C] or [1,H,W,C], float 0-1) -> base64 jpeg.

    Size and quality come from config.json "local" unless the caller overrides the
    size (the interrogator does - it is allowed to spend more on its single frame)."""
    from PIL import Image

    options = local_options()
    max_long_edge = max_long_edge or options["image_long_edge"]
    arr = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    w, h = pil.size
    scale = min(max_long_edge / max(w, h), 1.0)
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(options["jpeg_quality"]))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def strip_reasoning(text):
    """With thinking enabled the chat template prefills '<think>', so the reply
    arrives as '<reasoning></think>\\n\\n<answer>'; keep only the answer."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


# Qwen3.8 official sampling presets from the model card; thinking is the default mode.
# (llama-cpp-python spells presence_penalty as present_penalty.)
SAMPLING_THINKING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, present_penalty=0.0, repeat_penalty=1.0)
SAMPLING_NON_THINKING = dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, present_penalty=1.5, repeat_penalty=1.0)


def resolve_sampling(thinking, temperature=-1.0, top_p=-1.0, top_k=-1, presence_penalty=-1.0):
    """A negative widget value means "auto": follow the Qwen3.8 official preset for
    the current thinking mode. A manual value overrides just that parameter.
    min_p/repeat_penalty are pinned to the official values (0.0 / 1.0) in both modes."""
    preset = SAMPLING_THINKING if thinking else SAMPLING_NON_THINKING
    return {
        "temperature": temperature if temperature >= 0 else preset["temperature"],
        "top_p": top_p if top_p >= 0 else preset["top_p"],
        "top_k": top_k if top_k >= 0 else preset["top_k"],
        "min_p": preset["min_p"],
        "present_penalty": presence_penalty if presence_penalty >= 0 else preset["present_penalty"],
        "repeat_penalty": preset["repeat_penalty"],
        # llama-cpp only applies the penalties to the last N tokens; the Qwen presets
        # above assume the whole reply, so this is the knob that lines the two up.
        "penalty_last_n": local_options()["penalty_last_n"],
    }


def apply_text_template_args(llm, config):
    """Text-only loads get no Qwen35ChatHandler (it requires an mmproj), so the
    embedded chat template would render with its own defaults — for Qwen3.8 that
    means thinking always on. Rebind the template with the thinking kwargs baked
    in. Best effort: any failure keeps the previous behavior."""
    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        return
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        tmpl_args = {"enable_thinking": bool(config.get("thinking", False))}
        if config.get("reasoning_effort"):
            tmpl_args["reasoning_effort"] = config["reasoning_effort"]

        class _PresetFormatter(Jinja2ChatFormatter):
            def __call__(self, *, messages, **kwargs):
                return super().__call__(messages=messages, **{**tmpl_args, **kwargs})

        def _text(tid):
            return llm._model.token_get_text(tid) if tid != -1 else ""

        ids = {name: getattr(llm, f"token_{name}")() for name in ("eos", "eot", "sep", "nl", "pad")}
        stop_ids = [i for i in (ids["eos"], ids["eot"]) if i != -1] or None
        special = {f"{name}_token": t for name, i in ids.items() if (t := _text(i))}
        fmt = _PresetFormatter(
            template=template,
            eos_token=_text(ids["eos"]),
            bos_token=_text(llm.token_bos()),
            stop_token_ids=stop_ids,
            special_tokens_map=special,
        )
        llm.chat_handler = fmt.to_chat_handler()
        log(f"text-only load: chat template rebound with {tmpl_args}")
    except Exception as e:
        log(f"WARNING: text-only thinking control unavailable ({type(e).__name__}: {e}); template defaults apply")


class LocalLLM:
    llm = None
    chat_handler = None
    config = None

    # Nodes call this as LocalLLM.resolve_sampling(...) without a new import.
    resolve_sampling = staticmethod(resolve_sampling)

    @classmethod
    def unload(cls):
        if cls.llm is None and cls.chat_handler is None:
            return
        try:
            cls.llm.close()
        except Exception:
            pass
        try:
            cls.chat_handler.close()
        except Exception:
            pass
        try:
            cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        cls.llm = None
        cls.chat_handler = None
        cls.config = None
        gc.collect()
        mm.soft_empty_cache()
        log("local LLM unloaded")

    @classmethod
    def load(cls, config):
        if cls.llm is not None and cls.config == config:
            return
        cls.unload()
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen35ChatHandler

        mmproj = config.get("mmproj")
        handler = None
        if mmproj:
            log(f"loading mmproj: {mmproj}")
            handler = Qwen35ChatHandler(
                mmproj_path=mmproj,
                enable_thinking=config.get("thinking", False),
                image_min_tokens=config.get("image_min_tokens", -1),
                image_max_tokens=config.get("image_max_tokens", -1),
                verbose=False,
            )
            # Qwen3.8 thinking-depth knob; silently ignored by templates that don't define it
            if config.get("reasoning_effort"):
                handler.extra_template_arguments["reasoning_effort"] = config["reasoning_effort"]
        kwargs = {}
        if config.get("n_cpu_moe", 0) > 0:
            kwargs["n_cpu_moe"] = config["n_cpu_moe"]
        if config.get("kv_type"):
            kwargs["type_k"] = kwargs["type_v"] = resolve_kv_type(config["kv_type"])
        log(f"loading model: {config['model_path']} (n_gpu_layers={config['n_gpu_layers']}, n_ctx={config['n_ctx']}, n_cpu_moe={kwargs.get('n_cpu_moe', 0)}, kv_type={config.get('kv_type') or 'f16'}, image_tokens={config.get('image_min_tokens', -1)}..{config.get('image_max_tokens', -1)})")
        cls.llm = Llama(
            config["model_path"],
            chat_handler=handler,
            n_gpu_layers=config["n_gpu_layers"],
            n_ctx=config["n_ctx"],
            verbose=False,
            **kwargs,
        )
        if handler is None:
            apply_text_template_args(cls.llm, config)
        cls.chat_handler = handler
        cls.config = config.copy()

    @classmethod
    def has_vision(cls):
        return cls.config is not None and cls.config.get("mmproj") is not None

    @classmethod
    def clear_context(cls):
        """Drop KV between runs when the model stays loaded (Qwen3.5 hybrid cache included)."""
        if cls.llm is None:
            return
        cls.llm.n_tokens = 0
        cls.llm._ctx.memory_clear(True)
        if cls.llm.is_hybrid and cls.llm._hybrid_cache_mgr is not None:
            cls.llm._hybrid_cache_mgr.clear()

    @classmethod
    def generate(cls, messages, max_tokens, sampling, seed, on_text=None):
        """Full reply as one string; streams progress chunks to on_text when given.

        sampling: dict of create_chat_completion sampling kwargs, built by
        resolve_sampling() so it follows the thinking mode unless overridden."""
        log(f"sampling: {sampling}")
        # thinking mode: the chat template prefills '<think>\n' into the prompt, so the
        # streamed text starts mid-reasoning without the opening tag — restore it on the
        # display copy to mark which part is reasoning (downstream output stays clean).
        display_prefix = "<think>" if cls.config and cls.config.get("thinking") else ""
        thinking = bool(cls.config and cls.config.get("thinking"))
        chunks, last_push = [], 0.0
        t_start, t_first, pushes = time.time(), None, 0
        finish_reason = None
        for chunk in cls.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, seed=seed, stream=True, **sampling,
        ):
            choice = chunk["choices"][0]
            piece = choice.get("delta", {}).get("content")
            if piece:
                if t_first is None:
                    t_first = time.time()
                chunks.append(piece)
                if on_text and time.time() - last_push > 0.25:
                    last_push = time.time()
                    pushes += 1
                    on_text(display_prefix + "".join(chunks))
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        raw = "".join(chunks).strip()
        log(f"stream stats: first chunk +{((t_first or time.time()) - t_start):.1f}s, {len(chunks)} chunks, {pushes} pushes, total {time.time() - t_start:.1f}s, finish={finish_reason}")
        if on_text:
            on_text(display_prefix + raw)  # the node display keeps the full process, thinking included
        if finish_reason == "length":
            # A reply cut off at max_tokens (or the context limit) is incomplete: with
            # thinking on the reasoning eats the budget before the answer even starts,
            # with thinking off the answer itself is cut mid-sentence. Never hand back
            # half a prompt as if it were the result.
            raise ValueError(
                f"generation stopped at max_tokens={max_tokens} before finishing; thinking tokens share "
                "this budget, so raise it (-1 = no cap)"
            )
        if thinking and "</think>" not in raw:
            log("WARNING: thinking produced no </think> before the model stopped; passing the text through as-is")
        return strip_reasoning(raw)


def apply_unload_hook():
    """ComfyUI's 'free memory' should also drop the local LLM."""
    if getattr(mm, "_h3_unload_hook", False):
        return
    original = mm.unload_all_models

    def hooked(*args, **kwargs):
        LocalLLM.unload()
        return original(*args, **kwargs)

    mm.unload_all_models = hooked
    mm._h3_unload_hook = True
