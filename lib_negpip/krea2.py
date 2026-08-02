"""NegPiP support for Forge Neo's Krea 2 diffusion model.

Krea 2 encodes the prompt with Qwen3-VL, then concatenates the resulting text
tokens with the reference and image streams inside a single transformer.
NegPiP is applied by restoring the magnitude of negatively weighted text
embeddings, then negating the value projection of those tokens in every
single-stream block.

The text engine turns emphasis off entirely as soon as a reference image is
attached, since the vision tower expands one token into many embeddings and the
weights no longer line up.  That is why Edit mode used to ignore negative
weights, so the engine hooks below keep the emphasis alive for that path and
expand the weights over the inserted embeddings instead.
"""

import sys
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

import torch

from backend.args import dynamic_args
from backend.sampling import condition, sampling_function
from modules import shared

ENGINE_METHODS: tuple[str, ...] = ("__call__", "tokenize_line", "process_embeds")
ENGINE_REQUIREMENTS: tuple[str, ...] = ("process_tokens", "strip_template", "tokenize")


class _EngineState:
    """Carries the per-chunk data between the patched text engine methods."""

    engine_cls: type = None
    """the patched text engine class"""

    module = None
    """the module the text engine lives in"""

    multipliers: list[float] = None
    """emphasis weights of the chunk currently being encoded"""

    aligned: list[float] = None
    """^ the same weights, expanded over the embeddings of every image"""

    masks: list[torch.Tensor] = None
    """sign masks of the last engine call, one per prompt"""

    def reset(self):
        self.multipliers = None
        self.aligned = None
        self.masks = None


_state = _EngineState()
_originals: dict[str, Callable] = {}


def patch_krea2_negpip(cls: "NegPiP", *, unpatch: bool = False):
    """Install or remove all Krea 2 hooks.

    Hooks are stored on the objects they modify so model changes and failed or
    interrupted generations can safely call this function more than once.
    """
    if unpatch != cls._patched[2]:
        return

    model = getattr(cls, "_krea2_patched_model", None) if unpatch else shared.sd_model
    if model is None:
        cls._patched[2] = False
        return
    dit = model.forge_objects.unet.model.diffusion_model
    if unpatch:
        _unhook(model, dit)
        del cls._krea2_patched_model
        cls._patched[2] = False
        return

    # Validate the model before changing any global methods.
    engine = _find_text_engine(model)
    list(_single_blocks(dit))
    cls._krea2_patched_model = model
    try:
        _hook_engine(engine, False)
        _hook_get_learned_conditioning(model, False)
        _hook_dit_forward(dit, False)
        _hook_value_projections(dit, False)
        _hook_compile_conditions(False)
    except Exception:
        # Each remover tolerates a partially installed hook set.
        _unhook(model, dit)
        del cls._krea2_patched_model
        raise
    cls._patched[2] = True


def _unhook(model, dit):
    _hook_get_learned_conditioning(model, True)
    _hook_dit_forward(dit, True)
    _hook_value_projections(dit, True)
    _hook_compile_conditions(True)
    _hook_engine(None, True)
    _state.reset()


def _find_text_engine(model):
    for name in (
        "text_processing_engine_qwen",
        "text_processing_engine_qwen3_vl",
        "text_processing_engine_qwen3vl",
        "text_processing_engine_krea2",
        "text_processing_engine",
    ):
        engine = getattr(model, name, None)
        if engine is not None:
            return engine

    for name, engine in vars(model).items():
        if name.startswith("text_processing_engine") and engine is not None:
            return engine

    raise RuntimeError("NegPiP could not find Krea 2's Qwen3-VL text engine")


# ================================================================================ #
# Text Engine


def _hook_engine(engine, remove: bool):
    """Patch the text engine class, as `__call__` bypasses the instance."""

    if remove:
        for name, original in _originals.items():
            if getattr(_state.engine_cls, name, None) is not None:
                setattr(_state.engine_cls, name, original)
        _originals.clear()
        _state.engine_cls = None
        _state.module = None
        return

    engine_cls = type(engine)
    module = sys.modules.get(engine_cls.__module__, None)

    for attr in ENGINE_REQUIREMENTS:
        if not callable(getattr(engine, attr, None)):
            raise RuntimeError(f"NegPiP could not find Krea 2's {attr}")
    if getattr(module, "PromptChunk", None) is None:
        raise RuntimeError("NegPiP could not find Krea 2's PromptChunk")
    if not getattr(module, "KREA2_TAP_LAYERS", None):
        raise RuntimeError("NegPiP could not find Krea 2's hidden layers")

    _state.engine_cls = engine_cls
    _state.module = module

    replacements = {
        "__call__": _negpip_call,
        "tokenize_line": _negpip_tokenize_line,
        "process_embeds": _negpip_process_embeds,
    }

    for name in ENGINE_METHODS:
        _originals[name] = getattr(engine_cls, name)
        setattr(engine_cls, name, replacements[name])


def _negpip_call(self, texts: list[str], images: list[torch.Tensor] = []):
    """Encode every prompt, keeping the emphasis weights of the image path."""

    from backend.text_processing import emphasis

    self.emphasis = emphasis.get_current_option(shared.opts.emphasis)()

    if any(emphasis.uses_emphasis(x) for x in texts):
        dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

    # "None" and "Ignore" never scale the embeddings, leaving no negative
    # magnitude for NegPiP to restore
    weighted = isinstance(self.emphasis, emphasis.EmphasisOriginal)

    zs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    cache: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}

    for line in texts:
        if line not in cache:
            cache[line] = _encode_line(self, line, images, weighted)

        line_zs, line_masks = cache[line]
        zs.extend(line_zs)
        masks.extend(line_masks)

    _state.masks = masks
    return zs


def _encode_line(engine, line: str, images: list[torch.Tensor], weighted: bool):
    layers = len(_state.module.KREA2_TAP_LAYERS)

    zs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for chunk in engine.tokenize_line(line, images):
        tokens = chunk.tokens
        multipliers = list(chunk.multipliers)

        # the weights are expanded in place by `_negpip_process_embeds`, so
        # that the engine applies them to the embeddings of the image path too
        _state.multipliers = multipliers
        _state.aligned = None
        try:
            z = engine.process_tokens([tokens], [multipliers])
            aligned = _state.aligned
        finally:
            _state.multipliers = None
            _state.aligned = None

        z = engine.strip_template(z, tokens)
        b, seq, fuse = z.shape
        z = z.reshape(b * seq, layers, fuse // layers)

        zs.append(z)
        masks.append(_sign_mask(aligned if weighted else None, z))

    return zs, masks


def _negpip_tokenize_line(self, line: str, images: list[torch.Tensor] = []):
    """Tokenize every emphasis region, inserting the vision block only once."""

    from backend.text_processing import parsing

    parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
    if not parsed:
        parsed = [["", 1.0]]

    texts = [text for text, _ in parsed]
    weights = [weight for _, weight in parsed]

    if not images:
        tokenized = self.tokenize(texts)
    elif weights[0] == 1.0:
        # upstream prepends the vision block to every region, which duplicates
        # the reference image once per region of an emphasized prompt
        tokenized = self.tokenize(texts[:1], len(images))
        if len(texts) > 1:
            tokenized = tokenized + self.tokenize(texts[1:])
    else:
        # the leading region is emphasized, so the image gets a neutral one
        tokenized = self.tokenize([""], len(images)) + self.tokenize(texts)
        weights.insert(0, 1.0)

    chunk = _state.module.PromptChunk()
    embed_count = 0

    for tokens, weight in zip(tokenized, weights):
        for token in tokens:
            if token == self.id_image:
                token = {
                    "type": "image",
                    "data": images[embed_count],
                    "original_type": "image",
                }
                embed_count += 1

            chunk.tokens.append(token)
            chunk.multipliers.append(weight)

    return [chunk]


def _negpip_process_embeds(self, batch_tokens):
    result = _originals["process_embeds"](self, batch_tokens)

    multipliers = _state.multipliers
    if multipliers is not None and len(batch_tokens) == 1:
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("Unexpected Krea 2 embeddings")

        embeds, _, _, embeds_info = result
        aligned = _align_multipliers(
            batch_tokens[0], multipliers, embeds_info, embeds.shape[1]
        )
        if aligned is not None:
            _state.aligned = aligned
            # the engine only applies the emphasis when the weights match the
            # embeddings, which is exactly what the expansion above achieves
            multipliers[:] = aligned

    return result


def _align_multipliers(
    tokens: list,
    multipliers: list[float],
    embeds_info: Optional[list[dict]],
    length: int,
) -> Optional[list[float]]:
    """Expand the weights over the embeddings that every image turns into."""

    if len(tokens) != len(multipliers):
        return None

    weights = [
        w for token, w in zip(tokens, multipliers) if not isinstance(token, dict)
    ]
    images = [w for token, w in zip(tokens, multipliers) if isinstance(token, dict)]

    inserts = [
        info
        for info in (embeds_info or ())
        if isinstance(info, dict) and "index" in info and "size" in info
    ]

    if len(inserts) != len(images):
        # an embed was dropped, so which image was skipped is unknown; the
        # vision block is never emphasized anyway
        images = [1.0] * len(inserts)

    for weight, info in zip(images, inserts):
        index = int(info["index"])
        weights[index:index] = [weight] * int(info["size"])

    return weights if len(weights) == length else None


def _sign_mask(weights: Optional[list[float]], z: torch.Tensor) -> torch.Tensor:
    """Build the ±1 mask of a prompt, aligned with its conditioning."""

    length = z.shape[0]
    ones = torch.ones(length, 1, 1, device=z.device, dtype=z.dtype)

    if not weights or len(weights) < length:
        return ones

    # the template of the first region was stripped off the front
    values = torch.as_tensor(weights[-length:], device=z.device, dtype=z.dtype)
    return torch.where(values.reshape(length, 1, 1) < 0, -ones, ones)


# ================================================================================ #
# Conditioning


def _hook_get_learned_conditioning(model, remove: bool):
    if remove:
        original = getattr(model, "negpip_orig_get_learned_conditioning", None)
        if original is not None:
            if getattr(model.get_learned_conditioning, "_negpip_krea2", False):
                model.get_learned_conditioning = original
            del model.negpip_orig_get_learned_conditioning
        return

    original = model.get_learned_conditioning
    model.negpip_orig_get_learned_conditioning = original

    @torch.inference_mode()
    @wraps(original)
    def get_learned_conditioning(prompt):
        _state.masks = None
        conds = original(prompt)
        masks, _state.masks = _state.masks, None

        # Krea 2 returns one [tokens, layers, features] tensor per prompt.
        # Keep this deliberately strict: silently losing the mask is worse than
        # exposing an upstream conditioning API change.
        if not isinstance(conds, list) or len(conds) != len(prompt):
            raise RuntimeError("Unexpected Krea 2 conditioning result")
        if masks is None or len(masks) != len(conds):
            raise RuntimeError("Unexpected Krea 2 conditioning masks")

        contexts, negpip_masks = [], []
        count = 0

        for cond, mask in zip(conds, masks):
            if not isinstance(cond, torch.Tensor):
                raise RuntimeError("Unexpected Krea 2 conditioning item")

            mask = mask.to(cond)
            count += int((mask < 0).sum())
            contexts.append(cond * mask)
            negpip_masks.append(mask)

        if count:
            key = (
                "Negative"
                if getattr(prompt, "is_negative_prompt", False)
                else "Positive"
            )
            print(f"NegPiP Enable ({key}: {count})")

        # lists, so that prompts of different lengths are padded downstream
        return {"crossattn": contexts, "c_negpip_mask": negpip_masks}

    get_learned_conditioning._negpip_krea2 = True
    model.get_learned_conditioning = get_learned_conditioning


def _hook_compile_conditions(remove: bool):
    if remove:
        original = getattr(condition, "negpip_krea2_orig_compile_conditions", None)
        if original is not None:
            condition.compile_conditions = original
            sampling_function.compile_conditions = original
            del condition.negpip_krea2_orig_compile_conditions
        return

    original = condition.compile_conditions
    condition.negpip_krea2_orig_compile_conditions = original

    @wraps(original)
    def compile_conditions(cond):
        if isinstance(cond, dict) and "crossattn" in cond and "vector" not in cond:
            crossattn = cond["crossattn"]
            model_conds = {"c_crossattn": condition.ConditionCrossAttn(crossattn)}
            if "c_negpip_mask" in cond:
                model_conds["c_negpip_mask"] = condition.Condition(
                    cond["c_negpip_mask"]
                )
            return [dict(cross_attn=crossattn, model_conds=model_conds)]
        return original(cond)

    compile_conditions._negpip_krea2 = True
    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions


# ================================================================================ #
# Transformer


def _hook_dit_forward(dit, remove: bool):
    if remove:
        original = getattr(dit, "negpip_orig_forward", None)
        if original is not None:
            if getattr(dit.forward, "_negpip_krea2", False):
                dit.forward = original
            del dit.negpip_orig_forward
        if hasattr(dit, "_negpip_mask"):
            del dit._negpip_mask
        return

    original = dit.forward
    dit.negpip_orig_forward = original

    @torch.inference_mode()
    @wraps(original)
    def forward(*args, **kwargs):
        dit._negpip_mask = _attention_mask(kwargs.get("c_negpip_mask"))
        try:
            return original(*args, **kwargs)
        finally:
            dit._negpip_mask = None

    forward._negpip_krea2 = True
    dit.forward = forward


def _attention_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Reshape the conditioning mask to the [batch, tokens, 1] of attention."""

    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        return None

    batch, tokens = mask.shape[0], mask.shape[1]
    if mask.numel() != batch * tokens:
        return None
    if not bool((mask < 0).any()):
        return None

    return mask.reshape(batch, tokens, 1)


def _single_blocks(dit):
    blocks = None
    for name in ("blocks", "single_blocks", "transformer_blocks"):
        blocks = getattr(dit, name, None)
        if blocks is not None:
            break
    if blocks is None:
        raise RuntimeError("NegPiP could not find Krea 2 single-stream blocks")
    return blocks


def _hook_value_projections(dit, remove: bool):
    for block in _single_blocks(dit):
        attention = getattr(block, "attn", getattr(block, "attention", None))
        projection = getattr(attention, "wv", None)
        if projection is None:
            if remove:  # tolerate a partially installed hook set
                continue
            raise RuntimeError("NegPiP could not find a Krea 2 value projection")

        if remove:
            original = getattr(projection, "negpip_orig_forward", None)
            if original is not None:
                if getattr(projection.forward, "_negpip_krea2", False):
                    projection.forward = original
                del projection.negpip_orig_forward
            continue

        original = projection.forward
        projection.negpip_orig_forward = original

        @torch.inference_mode()
        @wraps(original)
        def forward(x, *args, __original=original, **kwargs):
            values = __original(x, *args, **kwargs)
            mask = getattr(dit, "_negpip_mask", None)
            if mask is None:
                return values
            if values.shape[0] % mask.shape[0]:
                raise RuntimeError("Krea 2 NegPiP mask batch does not match attention")
            if values.shape[0] != mask.shape[0]:
                mask = mask.repeat(values.shape[0] // mask.shape[0], 1, 1)
            # the prompt is always at the front of the reference and image streams
            text_length = min(values.shape[1], mask.shape[1])
            values[:, :text_length] *= mask[:, :text_length].to(values)
            return values

        forward._negpip_krea2 = True
        projection.forward = forward
