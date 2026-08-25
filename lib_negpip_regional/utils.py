import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from . import IS_NEO
from . import regions

NEG_PATTERN = re.compile(r"\(\s*(?:[^\\(:)]|\\[\(\)])+?\s*\:\s*-\s*\d*\.?\d+\s*\)")


def is_krea2(model) -> bool:
    """Whether this is Krea 2, without insisting on the name of its class.

    A rename or a repackaged build would otherwise drop straight through to
    doing nothing at all, which is indistinguishable from the Extension not
    being installed.  `KREA2_TAP_LAYERS` is specific to its text engine, where
    `text_processing_engine_qwen` on its own would also match Qwen-Image.
    """

    if type(model).__name__.lower() in ("krea2", "krea_2"):
        return True

    engine = getattr(model, "text_processing_engine_qwen", None)
    if engine is None:
        return False

    module = sys.modules.get(type(engine).__module__, None)
    return getattr(module, "KREA2_TAP_LAYERS", None) is not None


def reset_prompt_cache(p: "StableDiffusionProcessing"):
    c = 3 if IS_NEO else 2

    p.cached_c = [None] * c
    p.cached_uc = [None] * c
    if hasattr(p, "cached_hr_c"):
        p.cached_hr_c = [None] * c
        p.cached_hr_uc = [None] * c


def hr_dealer(p: "StableDiffusionProcessing") -> tuple[bool, bool]:
    return (
        bool(getattr(p, "hr_prompts", None)),
        bool(getattr(p, "hr_negative_prompts", None)),
    )


def has_negative(prompt: str) -> bool:
    return bool(re.search(NEG_PATTERN, prompt))


def have_negative(prompts: list[str]) -> bool:
    return any(has_negative(p) for p in prompts)


def any_negative(p: "StableDiffusionProcessing") -> bool:
    return any(
        [
            have_negative(p.prompts),
            have_negative(p.negative_prompts),
            have_negative(getattr(p, "hr_prompts", None) or ""),
            have_negative(getattr(p, "hr_negative_prompts", None) or ""),
        ]
    )


def has_regions(prompt: str) -> bool:
    return regions.split(prompt).regional


def have_regions(prompts) -> bool:
    return any(has_regions(p) for p in (prompts or ()))


def any_regional(p: "StableDiffusionProcessing") -> bool:
    """Whether any prompt of this generation confines a term to a box.

    Separate from :func:`any_negative` because a region is worth engaging for
    on its own: ``REGION 0.6 0 1 1 (a lighthouse:1.4)`` has no negative weight
    anywhere in it, and is exactly the same machinery pointed the other way.
    Gating on the minus sign alone would make half the feature unreachable.
    """

    return any(
        [
            have_regions(p.prompts),
            have_regions(p.negative_prompts),
            have_regions(getattr(p, "hr_prompts", None)),
            have_regions(getattr(p, "hr_negative_prompts", None)),
        ]
    )


def flatten(prompt: str) -> str:
    """A regional prompt as a plain one: the boxes dropped, the terms kept.

    For every model that is not Krea 2.  The `REGION` lines have to come out of
    the prompt one way or another -- left in, they are encoded as the words
    "region zero zero one nought point four", which is a worse outcome than
    either honouring them or ignoring them.  Keeping the terms and dropping the
    coordinates is the reading that loses least: `(man:-1)` still removes a man,
    it just removes him from the whole picture.
    """

    parsed = regions.split(prompt)
    return parsed.combined if parsed.regional else prompt


def flatten_prompts(p: "StableDiffusionProcessing"):
    """Rewrite every prompt of this generation with :func:`flatten`.

    In place, and only the batch's own lists: `all_prompts` is what the PNG
    metadata is written from, and a prompt that cannot be pasted back and
    reproduced is not much of a record.
    """

    for name in ("prompts", "negative_prompts", "hr_prompts", "hr_negative_prompts"):
        prompts = getattr(p, name, None)
        if isinstance(prompts, list):
            prompts[:] = [flatten(prompt) for prompt in prompts]
