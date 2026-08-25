"""What actually happened, written down, because the picture will not say.

A region that does not work looks exactly like a region that was never read.
Both are an image with the subject still in the box, and everything between the
prompt and the pixels -- how the line parsed, how many tokens it became, where
those tokens landed, which patches the box covers, how much of the image's
attention those tokens command -- is invisible from the outside.  Three rounds
of this have been spent reasoning backwards from photographs.

So this module writes the middle down.  It is off unless it is asked for, it
costs one extra attention on a sample of rows when it is on, and it answers, in
order, the questions somebody debugging a region actually has:

    did the line parse                  -> `prompt`
    what did it become                  -> `tokens`
    where did it land                   -> `conditioning`
    is the geometry the one I drew      -> `geometry`, with a map of the grid
    did the mask get applied            -> `stage`
    how much is the region worth        -> `attention`

The last one is the only one that can say *why* a region is weak rather than
broken, and it is the reason this module exists rather than a handful of print
statements.  NegPiP flips the sign of a term's value projection; how much that
changes the picture depends entirely on how much of each patch's attention goes
to that term in the first place.  If the answer is a fraction of a per cent, a
sign flip is a fraction of a per cent of a change, and no amount of weight on
the embedding fixes that -- it is a fact about the attention, and it is
measurable, and until now nobody had measured it.

Where it writes
---------------
``<this Extension's folder>/negpip_regional.log``, next to the script, so that
it is where somebody looking at the Extension will look.  Appended, rolled at
two megabytes, and every line also goes to the console so that a user who has
one open does not have to go and find the file.
"""

from __future__ import annotations

import datetime
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILENAME = "negpip_regional.log"
PATH = os.path.join(ROOT, FILENAME)

MAX_BYTES = 2 * 1024 * 1024
"""Rolled rather than grown: a log nobody prunes is a disk somebody loses."""

PREFIX = "NegPiP Regional"

SAMPLE = 64
"""How many image patches to measure the attention of, inside and out.

A mean over sixty-four patches of a region's attention share is the same number
to two decimal places as a mean over four thousand, and it is the difference
between a diagnostic that costs nothing and one somebody turns off.
"""

_enabled = False
_opened = False
_said: dict = {}
"""The last thing written under each heading, to keep a per-step call quiet.

`conditioning` is called from the plan, and the plan is rebuilt on every
forward -- thirty times an image -- because the patch grid it addresses is only
known once the latent for that step is in hand.  What it reports does not
change between those forwards, except across a highres pass, which is exactly
when it should be said again.  So it is deduplicated on its own content rather
than counted.
"""


def enable(on: bool):
    global _enabled
    _enabled = bool(on)


def enabled() -> bool:
    return _enabled


def _roll():
    """Start a new file when the old one is big, keeping one generation back."""
    try:
        if os.path.getsize(PATH) < MAX_BYTES:
            return
        backup = PATH + ".1"
        if os.path.exists(backup):
            os.remove(backup)
        os.replace(PATH, backup)
    except OSError:
        pass


def say(line: str = ""):
    """One line, to the console and to the file. Never raises.

    Never raises because a diagnostic that can stop a generation is worse than
    no diagnostic: a read-only folder, a file open in an editor with a lock on
    it, a path on a drive that has gone away -- each of those is a line that
    does not get written, and a picture that still gets made.
    """

    if not _enabled:
        return

    print(f"{PREFIX}: {line}" if line else "")

    global _opened
    try:
        if not _opened:
            _roll()
            _opened = True
        with open(PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def begin(where: str = ""):
    """Head the log with the time, so two runs cannot be read as one."""
    if not _enabled:
        return
    _said.clear()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    say("")
    say(f"=== {stamp} {where} ".ljust(72, "="))


# ================================================================================ #
# The prompt, and what it became


def prompt(parsed, which: str = ""):
    """The REGION lines as they parsed. The first thing that can be wrong."""

    if not _enabled:
        return

    if not parsed.regions:
        say(f"prompt{which}: no REGION lines")
        return

    say(f"prompt{which}: {len(parsed.regions)} region(s) parsed")
    for number, region in enumerate(parsed.regions, 1):
        x0, y0, x1, y1 = region.box
        say(f"  region {number}  box ({x0:.3f}, {y0:.3f}, {x1:.3f}, {y1:.3f})"
            f"  text {region.text!r}")
    say(f"  scene: {parsed.scene!r}")


def tokens(boxes, weights, length: int):
    """Which tokens each box owns, and what weight they carry.

    The weight is the half that is easy to lose.  A region whose tokens carry
    ``1.0`` parsed as a box but not as a negative -- the emphasis mode ate the
    weight, or the fragment was written in a way `parse_prompt_attention` reads
    as literal text -- and the box is then perfectly applied to a term that is
    not negative at all.
    """

    if not _enabled or not boxes:
        return

    tail = list(boxes[-length:]) if len(boxes) >= length else list(boxes)
    signs = list(weights[-length:]) if weights and len(weights) >= length else []

    runs = []
    for index, box in enumerate(tail):
        if not box:
            continue
        if runs and runs[-1][2] == index and runs[-1][0] == box:
            runs[-1][2] = index + 1
        else:
            runs.append([box, index, index + 1])

    if not runs:
        say(f"tokens: {length} conditioning token(s), none carrying a region")
        return

    for number, (box, start, stop) in enumerate(runs, 1):
        piece = signs[start:stop] if signs else []
        weight = (f", weight {min(piece):+.2f}..{max(piece):+.2f}"
                  if piece else "")
        say(f"tokens: region {number} -> {stop - start} token(s) "
            f"at {start}..{stop} of {length}{weight}")
        if piece and min(piece) >= 0:
            say("  ^ nothing negative here: the box will confine the term, "
                "but there is no sign to apply inside it")


def conditioning(spans, txtlen: int):
    """Where the regions ended up once the batch was stacked and padded."""

    if not _enabled:
        return

    lines = [f"conditioning: {len(spans)} prompt(s), {txtlen} text token(s)"]
    for number, row in enumerate(spans, 1):
        if not row:
            lines.append(f"  prompt {number}: no regions")
            continue
        for span in row:
            x0, y0, x1, y1 = span.box
            lines.append(f"  prompt {number}: [{span.start}..{span.stop}] "
                         f"box ({x0:.3f}, {y0:.3f}, {x1:.3f}, {y1:.3f})")

    if _said.get("conditioning") == lines:
        return
    _said["conditioning"] = lines

    for line in lines:
        say(line)


# ================================================================================ #
# The geometry, drawn


def geometry(plan, total: int):
    """The patch grid, and a picture of where each box landed on it.

    Drawn rather than described because "rows 0..32, columns 0..64" is a thing
    somebody has to hold in their head and compare against a rectangle they
    drew ten minutes ago, and a shape on the screen is a thing they can see is
    wrong.  It is also the check that catches the one silent failure that
    matters -- a mask built for a resolution other than the one being sampled.
    """

    if not _enabled:
        return

    grid = plan.geometry
    say(f"geometry: grid {grid.height}x{grid.width} = {grid.imglen} patches, "
        f"{grid.reflen(total)} reference token(s), {total} in the stream")

    from . import regions as _regions

    for row in plan.spans:
        for number, span in enumerate(row, 1):
            top, left, bottom, right = _regions.patch_bounds(
                span.box, grid.height, grid.width)
            covered = (bottom - top) * (right - left)
            say(f"  region {number} covers {covered} of {grid.imglen} patches, "
                f"rows {top}..{bottom}, columns {left}..{right}")
            for line in _map(top, left, bottom, right, grid.height, grid.width):
                say("    " + line)
        break  # one prompt's worth; the rest are the same boxes or none


def _map(top, left, bottom, right, height, width, columns=48, rows=16):
    """The grid as characters, downsampled to something a console can hold."""

    drawn = []
    for row in range(rows):
        y = int(row * height / rows)
        line = ""
        for column in range(columns):
            x = int(column * width / columns)
            line += "#" if (top <= y < bottom and left <= x < right) else "."
        drawn.append(line)
    return drawn


# ================================================================================ #
# What the mask was worth


def stage(name: str, detail: str = ""):
    say(f"stage: {name}{(' -- ' + detail) if detail else ''}")


def attention(q, k, v, plan, total: int, scale=None):
    """How much of each patch's attention the region's tokens actually command.

    This is the number that decides whether a region can work at all.  NegPiP
    changes the *sign* of what a term contributes; the size of that change is
    the share of attention the term already had.  A term holding half a per
    cent of a patch's attention, flipped, moves that patch by half a per cent,
    and a picture made of such changes looks like the prompt was ignored.

    Measured on a sample of patches inside the box and a sample outside it, on
    the tensors of one real block of one real step, with the log-sum-exp over
    every key as the denominator -- so it is the true softmax share and not a
    proxy for it.  The outside number is what the mask took away; the inside
    number is what the sign has to work with.
    """

    if not _enabled:
        return

    from .regional import _lse_attention, patch_rows

    grid = plan.geometry
    start = grid.txtlen + grid.reflen(total)

    for item, row in enumerate(plan.spans):
        for number, span in enumerate(row, 1):
            inside = patch_rows(span.box, grid.height, grid.width, q.device)
            rows_in = (inside.nonzero().flatten() + start)
            rows_out = ((~inside).nonzero().flatten() + start)

            shares = []
            for label, rows in (("inside the box ", rows_in),
                                ("outside the box", rows_out)):
                if rows.numel() == 0:
                    shares.append((label, None))
                    continue
                picked = rows[torch.linspace(
                    0, rows.numel() - 1, min(SAMPLE, rows.numel()),
                    device=rows.device).long()]
                shares.append((label, _share(
                    q, k, v, item, picked, span, scale, _lse_attention)))

            say(f"attention on region {number}:")
            for label, value in shares:
                if value is None:
                    say(f"  {label}: no patches")
                elif value < 0:
                    say(f"  {label}: could not be measured on this build")
                else:
                    kept = "kept" if "inside" in label else "removed by the mask"
                    say(f"  {label}: {value * 100:.3f}% of each patch's "
                        f"attention ({kept})")
        break


def _share(q, k, v, item, rows, span, scale, lse_attention) -> float:
    """The softmax share of ``span``'s keys, for the given query rows.

    Computed from the tensors as they are, with the region's keys still in
    them: what is wanted is what the region would be worth unmasked, so that
    the inside and the outside numbers are the same measurement and the mask is
    the only difference between them.
    """

    import math

    try:
        queries = q[item:item + 1][:, :, rows]
        keys = k[item:item + 1]
        values = v[item:item + 1]

        found = lse_attention(queries, keys, values, scale)
        factor = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(
            queries.float(),
            keys[:, :, span.start:span.stop].float().transpose(-1, -2)) * factor
        region = torch.logsumexp(scores, dim=-1)

        if found is None:
            # no log-sum-exp on this build, so the denominator is computed the
            # expensive way -- on sixty-four rows, which is affordable exactly
            # because it is sixty-four and not the whole picture
            whole = torch.matmul(
                queries.float(), keys.float().transpose(-1, -2)) * factor
            total = torch.logsumexp(whole, dim=-1)
        else:
            total = found[1]

        return float(torch.exp(region - total).mean())
    except Exception:
        return -1.0
