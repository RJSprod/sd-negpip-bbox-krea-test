"""NegPiP support for Forge Neo's Krea 2 diffusion model.

Krea 2 concatenates prompt tokens with the image stream.  NegPiP is applied by
restoring the magnitude of negatively weighted text embeddings and negating the
corresponding value projection in each single-stream transformer block.
"""

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

import torch
import torch.nn.functional as F

from backend.sampling import condition, sampling_function
from modules import shared


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
        _hook_get_learned_conditioning(model, True)
        _hook_dit_forward(dit, True)
        _hook_value_projections(dit, True)
        _hook_compile_conditions(True)
        del cls._krea2_patched_model
        cls._patched[2] = False
        return

    # Validate the model before changing any global methods.
    _find_text_engine(model)
    list(_single_blocks(dit))
    cls._krea2_patched_model = model
    try:
        _hook_get_learned_conditioning(model, False)
        _hook_dit_forward(dit, False)
        _hook_value_projections(dit, False)
        _hook_compile_conditions(False)
    except Exception:
        # Each remover tolerates a partially installed hook set.
        _hook_get_learned_conditioning(model, True)
        _hook_dit_forward(dit, True)
        _hook_value_projections(dit, True)
        _hook_compile_conditions(True)
        del cls._krea2_patched_model
        raise
    cls._patched[2] = True


def _find_text_engine(model):
    for name in (
        "text_processing_engine_qwen3_vl",
        "text_processing_engine_qwen3vl",
        "text_processing_engine_krea2",
        "text_processing_engine",
    ):
        engine = getattr(model, name, None)
        if engine is not None:
            return engine
    raise RuntimeError("NegPiP could not find Krea 2's Qwen3-VL text engine")


def _multipliers(value):
    """Yield emphasis multipliers from Forge tokenizer return values."""
    for name in ("qwen_multipliers", "qwen3_vl_multipliers", "multipliers"):
        found = getattr(value, name, None)
        if found is not None:
            yield from found
            return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _multipliers(item)


def _build_negpip_mask(engine, line, length, device, dtype):
    weights = list(_multipliers(engine.tokenize_line(line)))
    if not weights:
        return torch.ones(length, device=device, dtype=dtype)

    weights = torch.as_tensor(weights, device=device, dtype=dtype).flatten()
    mask = torch.where(weights < 0, -torch.ones_like(weights), torch.ones_like(weights))
    if mask.numel() < length:
        mask = F.pad(mask, (0, length - mask.numel()), value=1.0)
    return mask[:length]


def _hook_get_learned_conditioning(model, remove):
    if remove:
        original = getattr(model, "negpip_orig_get_learned_conditioning", None)
        if original is not None:
            if getattr(model.get_learned_conditioning, "_negpip_krea2", False):
                model.get_learned_conditioning = original
            del model.negpip_orig_get_learned_conditioning
        return

    original = model.get_learned_conditioning
    model.negpip_orig_get_learned_conditioning = original
    engine = _find_text_engine(model)

    @torch.inference_mode()
    @wraps(original)
    def get_learned_conditioning(prompt):
        conds = original(prompt)
        # Krea 2 currently returns one [tokens, features] tensor per prompt.
        # Keep this deliberately strict: silently losing the mask is worse than
        # exposing an upstream conditioning API change.
        if not isinstance(conds, list) or len(conds) != len(prompt):
            raise RuntimeError("Unexpected Krea 2 conditioning result")

        contexts, masks = [], []
        count = 0
        for line, cond in zip(prompt, conds):
            if not isinstance(cond, torch.Tensor):
                raise RuntimeError("Unexpected Krea 2 conditioning item")
            context = cond.reshape(-1, cond.shape[-1])
            mask = _build_negpip_mask(
                engine, line, context.shape[0], context.device, context.dtype
            )
            count += int((mask < 0).sum().item())
            contexts.append(context * mask.unsqueeze(-1))
            masks.append(mask.unsqueeze(-1))

        if count:
            key = "Negative" if prompt.is_negative_prompt else "Positive"
            print(f"NegPiP Enable ({key}: {count})")
        return {
            "crossattn": torch.stack(contexts),
            "c_negpip_mask": torch.stack(masks),
        }

    get_learned_conditioning._negpip_krea2 = True
    model.get_learned_conditioning = get_learned_conditioning


def _hook_dit_forward(dit, remove):
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
        dit._negpip_mask = kwargs.get("c_negpip_mask")
        try:
            return original(*args, **kwargs)
        finally:
            dit._negpip_mask = None

    forward._negpip_krea2 = True
    dit.forward = forward


def _single_blocks(dit):
    blocks = None
    for name in ("single_blocks", "transformer_blocks", "blocks"):
        blocks = getattr(dit, name, None)
        if blocks is not None:
            break
    if blocks is None:
        raise RuntimeError("NegPiP could not find Krea 2 single-stream blocks")
    return blocks


def _hook_value_projections(dit, remove):
    for block in _single_blocks(dit):
        attention = getattr(block, "attention", getattr(block, "attn", None))
        projection = getattr(attention, "wv", None)
        if projection is None:
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
            text_length = min(values.shape[1], mask.shape[1])
            values[:, :text_length] *= mask[:, :text_length].to(values)
            return values

        forward._negpip_krea2 = True
        projection.forward = forward


def _hook_compile_conditions(remove):
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
                model_conds["c_negpip_mask"] = condition.Condition(cond["c_negpip_mask"])
            return [dict(cross_attn=crossattn, model_conds=model_conds)]
        return original(cond)

    compile_conditions._negpip_krea2 = True
    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions
