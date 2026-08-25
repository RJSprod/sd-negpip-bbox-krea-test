"""Where a weighted term applies, written in the prompt.

NegPiP gives a term a sign.  This module gives it an *address*: a rectangle of
the image the term is allowed to reach, so that

    a wide empty landscape at dawn
    REGION 0 0 1 0.4 (man:-1)

subtracts "man" from the top four tenths of the frame and leaves the rest of
the picture free to contain one.  Without an address, a negative weight is a
statement about the whole image and there is no way to say "not here".

The syntax
----------
One region per line, at the start of the line::

    REGION x0 y0 x1 y1  <terms>

The four numbers are the corners of the box as fractions of the image, left,
top, right, bottom, in the reading order of a CSS rectangle.  Fractions rather
than pixels because the prompt outlives the resolution it was written at: the
same prompt has to mean the same thing at 1024 and at 1536, and highres fix
runs the second pass at a different size than the first.

`<terms>` is an ordinary prompt fragment, weights and all.  Everything NegPiP
already understands keeps working inside a region -- ``(man:-1)`` subtracts,
``(man:10)`` insists -- and a term with no weight at all is simply a phrase
that only applies inside the box.  Commas, parentheses and LoRA tags are passed
through untouched, because the fragment is handed to the same encoder the rest
of the prompt goes to.

What this module does and does not do
-------------------------------------
It is pure text and arithmetic: no torch, no model, nothing that has to wait
for a generation.  :func:`split` takes a prompt and hands back the prompt with
the REGION lines removed plus the list of regions it found, and
:func:`token_span` and :func:`patch_mask` do the index arithmetic that turns a
region into the two things attention needs -- which text tokens are its, and
which image patches are inside its box.  The tensors are built in
:mod:`.regional`, from these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KEYWORD = "REGION"
"""What starts a region line.

Upper case and a whole word, in the idiom the prompt language already uses for
``BREAK``, ``AND``, ``ADDCOL``: those are the words a prompt parser reserves,
and a user who has met one has met the convention.  Deliberately not a bracket
or a brace -- ``[`` and ``(`` are already attention syntax, and ``{`` belongs
to Dynamic Prompts.  There is no character left that is free in every install.
"""

PATTERN = re.compile(
    r"^[ \t]*" + KEYWORD + r"[ \t]+"
    r"(-?\d*\.?\d+)[ \t,]+(-?\d*\.?\d+)[ \t,]+(-?\d*\.?\d+)[ \t,]+(-?\d*\.?\d+)"
    r"[ \t]*:?[ \t]*(.*)$",
    re.MULTILINE,
)
"""A region line.

Anchored to the start of a line so a prompt that happens to contain the word in
running text is left alone.  Separators are spaces or commas, because both are
what people type, and an optional colon after the numbers reads well without
being required.
"""


@dataclass
class Region:
    """One box and the prompt fragment that applies inside it."""

    box: tuple[float, float, float, float]
    """``(x0, y0, x1, y1)`` as fractions of the image, already ordered and
    clamped by :func:`_box`."""

    text: str
    """The prompt fragment, exactly as typed."""

    start: int = 0
    """Index of this region's first token in the conditioning. Filled in during
    encoding -- see :func:`token_span`."""

    length: int = 0
    """How many tokens the fragment became."""

    @property
    def empty(self) -> bool:
        return not self.text.strip() or self.length <= 0

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class Prompt:
    """A prompt taken apart into the scene and its regions."""

    scene: str = ""
    """The prompt with every REGION line removed. What everything sees."""

    regions: list[Region] = field(default_factory=list)

    @property
    def regional(self) -> bool:
        return bool(self.regions)

    @property
    def combined(self) -> str:
        """The scene followed by every region's text, which is what is encoded.

        The regions are appended rather than encoded separately so they share
        one pass of the text fusion transformer with the scene, and so their
        tokens land in one contiguous block whose position this module can then
        work out.  Encoding them apart would cost a second forward each and
        would still have to be concatenated here.
        """
        parts = [self.scene.strip()]
        parts += [region.text.strip() for region in self.regions]
        return ", ".join(part for part in parts if part)


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _box(x0: str, y0: str, x1: str, y1: str) -> tuple[float, float, float, float]:
    """Four strings as an ordered, clamped rectangle in 0..1.

    Corners are sorted rather than rejected: ``REGION 1 1 0 0`` is a rectangle
    somebody typed from the other corner, not an error worth refusing a
    generation over.  Values outside the frame are clamped for the same reason
    -- a box drawn slightly off the edge means the edge.
    """

    left, right = sorted((_number(x0), _number(x1)))
    top, bottom = sorted((_number(y0), _number(y1)))
    clamp = lambda v: 0.0 if v < 0.0 else 1.0 if v > 1.0 else v  # noqa: E731
    return (clamp(left), clamp(top), clamp(right), clamp(bottom))


def split(prompt: str) -> Prompt:
    """Take a prompt apart into its scene and its regions.

    A region whose box has no area is dropped: it addresses no patch, so
    everything downstream would treat it as text that reaches nothing, and
    silently encoding tokens that can never be attended to is the kind of
    no-op this fork exists to avoid.
    """

    text = str(prompt or "")
    if KEYWORD not in text:
        return Prompt(scene=text)

    regions: list[Region] = []

    for match in PATTERN.finditer(text):
        box = _box(*match.group(1, 2, 3, 4))
        region = Region(box=box, text=match.group(5).strip())
        if region.text and region.area > 0.0:
            regions.append(region)

    scene = PATTERN.sub("", text)
    # the removed lines leave their newlines behind
    scene = re.sub(r"\n{2,}", "\n", scene).strip()

    return Prompt(scene=scene, regions=regions)


def strip(prompt: str) -> str:
    """The prompt as everything that is not this Extension should see it."""
    return split(prompt).scene


def token_span(scene_length: int, lengths: list[int]) -> list[tuple[int, int]]:
    """Where each region's tokens are, given the scene's length and their own.

    The conditioning is the scene followed by the regions in order, so this is
    a running sum -- but it is written out because getting it wrong shifts a
    region's mask onto its neighbour's tokens, which looks like the box being
    in the wrong place rather than like an off-by-one.
    """

    spans: list[tuple[int, int]] = []
    cursor = max(0, int(scene_length))
    for length in lengths:
        length = max(0, int(length))
        spans.append((cursor, length))
        cursor += length
    return spans


def patch_bounds(box, height: int, width: int) -> tuple[int, int, int, int]:
    """A fractional box as ``(top, left, bottom, right)`` in patch coordinates.

    Krea 2's image stream is a row-major grid of ``height x width`` patches, so
    a box becomes a rectangle of grid cells.  Rounding is outward -- floor the
    near edge, ceil the far one -- so a box always covers at least the patches
    it touches, and a thin box never rounds away to nothing.  A region that
    addressed no patch would be a prompt fragment with no effect, reported by
    nobody.
    """

    import math

    x0, y0, x1, y1 = box
    top = int(math.floor(y0 * height))
    left = int(math.floor(x0 * width))
    bottom = int(math.ceil(y1 * height))
    right = int(math.ceil(x1 * width))

    top = max(0, min(top, height - 1))
    left = max(0, min(left, width - 1))
    bottom = max(top + 1, min(bottom, height))
    right = max(left + 1, min(right, width))

    return (top, left, bottom, right)


def patch_indices(box, height: int, width: int) -> list[int]:
    """The flat indices, into the image stream, of the patches inside ``box``.

    Row-major, matching the ``b (h w) c`` the transformer rearranges into --
    the same order the model's own position ids are built in, so index ``i``
    here is the patch at ``(i // width, i % width)`` there.
    """

    top, left, bottom, right = patch_bounds(box, height, width)
    return [
        row * width + column
        for row in range(top, bottom)
        for column in range(left, right)
    ]
