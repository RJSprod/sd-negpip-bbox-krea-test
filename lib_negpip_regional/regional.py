"""Confining a weighted term to a box, inside Krea 2's attention.

NegPiP carries a term's sign in the value projection: the concept is encoded at
its magnitude so attention still routes to it, and every single-stream block
multiplies the value of those tokens by a negative number, so the image stream
subtracts the concept instead of adding it.  That is a statement about the
whole picture, because every image token attends to every text token.

Regional NegPiP adds the other half: **which** image tokens are allowed to
attend to a term at all.  Take that away from the tokens outside a box and the
term stops existing for them -- it is neither added nor subtracted there -- and
the sign then applies only inside.  ``REGION 0 0 1 0.4 (man:-1)`` is exactly
that: no man in the top four tenths, and the rest of the frame untouched.

Why this needs an attention mask
--------------------------------
The sign mask is indexed by *key*: one number per text token, the same for
every query.  "Only here" is a statement about the query -- the same key must
be visible to one image patch and invisible to its neighbour -- so no per-key
quantity can express it, however it is weighted.  The query axis is the whole
difference between NegPiP and this fork.

Krea 2 makes the geometry available for free.  The sequence its single-stream
blocks attend over is ``[context | refs | img]``, laid out in that order with
known lengths, and the image half is a row-major ``h x w`` grid of patches --
the same grid the model builds its own position ids from.  So a box in
fractions of the image is a set of rows of the attention matrix, by arithmetic,
with nothing to infer and nothing to learn.

Two ways to apply it
--------------------
``dense`` builds the mask the obvious way: an additive ``[B, 1, L, L]`` bias,
``-inf`` wherever a query outside the box meets one of the region's keys, and
zero everywhere else.  It is about fifteen lines, it works on every attention
backend that takes a mask, and it is the definition the other path is checked
against.  It also costs ``L**2`` numbers, and ``L`` is around 9,500 at
1536x1536 -- 360 MB of mask, most of it structurally zero.

``merge`` never builds it.  The region's keys are a contiguous tail of the text
block, so the sequence splits into a global part (everything else, the
overwhelming majority) and one small part per region.  Attention over each part
is computed separately and the two are combined exactly, the way flash
attention combines its own chunks: keep each part's log-sum-exp, and

    out = (out_g * e**lse_g + out_r * e**lse_r) / (e**lse_g + e**lse_r)

is the softmax over the union.  Excluding a query from a region is then setting
that query's ``lse_r`` to ``-inf`` -- a vector of length ``L``, not a matrix --
and the region's own attention is over a handful of keys, so it costs a few
tens of megabytes and a few percent of the step.  The global attention runs on
the fast kernel it always ran on, with no mask at all.

``merge`` is the default where the log-sum-exp can be had from the attention
kernel, which is what :func:`_lse_attention` probes for once.  Everywhere else,
and whenever the two are being compared, ``dense`` is the fallback, and the
answers agree to floating-point noise -- which is what ``tests/`` asserts.

One patch point
---------------
``backend/nn/krea.py`` calls its blocks as ``block(combined, tvec, freqs, None,
...)``: the mask parameter exists the whole way down to ``attention_function``
and is hard-coded to ``None``, so there is nothing to pass a mask *into*.
Rather than wrap every block and every attention module, this replaces the
``attention_function`` name in that module with :func:`attend`, which reads the
plan for the forward that is running.  One name, restored on unpatch, and the
text fusion blocks -- whose sequence is the text alone -- fall out of it by
length, since their ``L`` cannot contain an image grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import regions

COLUMNS = 5
"""Width of the per-token region table: a flag and four coordinates.

The plan has to reach the transformer through the same conditioning machinery
the sign mask uses, which batches and pads tensors shaped ``[tokens, ...]``.  A
side table of boxes does not survive that; a column per coordinate, repeated on
every token the box owns, does -- and padding a prompt to its neighbour's length
adds rows whose flag is zero, which is already the right answer.
"""

MODES = ("auto", "merge", "dense")


@dataclass
class Geometry:
    """What one DiT forward knows about its own sequence."""

    txtlen: int = 0
    height: int = 0
    width: int = 0

    @property
    def imglen(self) -> int:
        return self.height * self.width

    def reflen(self, total: int) -> int:
        """Reference tokens, as the part of ``L`` that is neither text nor image."""
        return int(total) - self.txtlen - self.imglen

    def holds(self, total: int) -> bool:
        """Whether a sequence of ``total`` tokens is the combined stream.

        The text fusion blocks attend over the prompt alone and reach the same
        patched function; their ``L`` is the text length, which cannot also
        contain an image grid.
        """
        return self.imglen > 0 and self.reflen(total) >= 0 and total > self.txtlen


@dataclass
class Span:
    """One region, resolved against the sequence being attended over."""

    start: int
    length: int
    box: tuple[float, float, float, float]

    @property
    def stop(self) -> int:
        return self.start + self.length


@dataclass
class Plan:
    """Every prompt's regions for one forward, plus the geometry they index."""

    geometry: Geometry = field(default_factory=Geometry)
    spans: list[list[Span]] = field(default_factory=list)
    """One list per batch item, in the order the batch is stacked."""

    mode: str = "auto"

    @property
    def active(self) -> bool:
        return any(self.spans)


_plan: Optional[Plan] = None
"""The plan of the forward currently running, or None.

Module level because the seam is a module-level name: :func:`attend` replaces
``backend.nn.krea.attention_function``, and is called by every block of the
forward that set this.  Sampling is one forward at a time on one thread, which
is the same assumption the sign mask upstream already makes.
"""


def begin(plan: Optional[Plan]):
    """Make ``plan`` the one :func:`attend` reads. Called from the DiT forward."""
    global _plan
    _plan = plan if (plan is not None and plan.active) else None


def end():
    begin(None)


# ================================================================================ #
# Reading the plan out of the conditioning


def spans_from_table(table: Optional[torch.Tensor], txtlen: int) -> list[list[Span]]:
    """Turn the ``[B, T, 5]`` region table back into spans, per batch item.

    Tokens of one region are contiguous and carry identical coordinates, so a
    run of equal rows is a region.  Comparing the coordinates rather than
    trusting an index means two regions that happen to be adjacent stay two
    regions only if their boxes differ -- and if they do not differ, they are
    the same box and merging them is right.
    """

    if not isinstance(table, torch.Tensor) or table.ndim != 3:
        return []
    if table.shape[-1] != COLUMNS:
        return []

    rows: list[list[Span]] = []
    limit = min(int(txtlen), int(table.shape[1]))

    for item in table[:, :limit].float().cpu():
        spans: list[Span] = []
        for index in range(item.shape[0]):
            if float(item[index, 0]) <= 0.0:
                continue
            box = tuple(round(float(v), 6) for v in item[index, 1:])
            if spans and spans[-1].stop == index and spans[-1].box == box:
                spans[-1].length += 1
            else:
                spans.append(Span(start=index, length=1, box=box))
        rows.append(spans)

    return rows


def table_from_regions(regions, lengths, scene_length: int, total: int,
                       device=None, dtype=None) -> torch.Tensor:
    """Build the ``[T, 5]`` table one prompt's conditioning travels with."""

    table = torch.zeros(int(total), COLUMNS, device=device, dtype=dtype or torch.float32)
    cursor = int(scene_length)

    for region, length in zip(regions, lengths):
        length = int(length)
        if length <= 0:
            continue
        stop = min(cursor + length, int(total))
        if cursor < stop:
            table[cursor:stop, 0] = 1.0
            table[cursor:stop, 1:] = torch.tensor(
                list(region.box), device=table.device, dtype=table.dtype)
        cursor += length

    return table


# ================================================================================ #
# Geometry: a box as rows of the attention matrix


def patch_rows(box, height: int, width: int, device) -> torch.Tensor:
    """A boolean ``[height * width]`` of the image patches inside ``box``.

    Row-major, matching the ``b (h w) c`` the transformer rearranges its latent
    into, so index ``i`` is the patch at ``(i // width, i % width)`` -- the same
    order ``_imgids`` numbers them in.
    """

    top, left, bottom, right = regions.patch_bounds(box, height, width)

    inside = torch.zeros(height, width, dtype=torch.bool, device=device)
    inside[top:bottom, left:right] = True
    return inside.reshape(-1)


def query_rows(span: Span, geometry: Geometry, total: int, device) -> torch.Tensor:
    """A boolean ``[total]``: the queries allowed to see this region's keys.

    Only image patches inside the box, and deliberately nothing else.  Letting
    the scene's own text tokens attend to a region would put the concept back
    into the tokens every patch in the picture reads, which is the leak that
    makes a regional negative look like it did nothing: the sign is subtracted
    inside the box and the concept arrives everywhere by the side door.
    Reference tokens are excluded for the same reason.
    """

    allowed = torch.zeros(int(total), dtype=torch.bool, device=device)
    start = geometry.txtlen + geometry.reflen(total)
    inside = patch_rows(span.box, geometry.height, geometry.width, device)
    allowed[start:start + inside.shape[0]] = inside
    # a region's own tokens keep reading the prompt they are part of, so that
    # the fragment is encoded in context rather than in isolation
    allowed[span.start:span.stop] = True
    return allowed


# ================================================================================ #
# The two implementations


def _neutral(dtype: torch.dtype) -> float:
    """The additive bias that means "never".

    ``-inf`` is correct and is what the mask means, but an ``-inf`` that meets a
    zero-length key block or a fused kernel's padding produces ``NaN`` for the
    whole row, and a ``NaN`` at step three of thirty is a black image with no
    error attached.  The smallest finite number of the dtype is ``e**-65504``
    away from the alternatives in fp16, which is zero in any arithmetic that
    follows.
    """
    return torch.finfo(dtype).min


def dense_mask(plan: Plan, total: int, batch: int, device, dtype) -> torch.Tensor:
    """The ``[B, 1, L, L]`` additive bias. The definition; see the docstring."""

    bias = torch.zeros(batch, 1, total, total, device=device, dtype=dtype)
    blocked = _neutral(dtype)

    for index in range(batch):
        spans = plan.spans[index % len(plan.spans)] if plan.spans else []
        for span in spans:
            if span.length <= 0:
                continue
            bias[index, 0, :, span.start:span.stop] = blocked
            allowed = query_rows(span, plan.geometry, total, device)
            bias[index, 0, allowed, span.start:span.stop] = 0.0

    return bias


LSE_OPS = (
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
)
"""The private ops that hand back a log-sum-exp, best first.

``scaled_dot_product_attention`` computes the normaliser and throws it away, so
the merge has to reach one level below the public function.  Which of these
exists, and which one accepts the device and dtype in play, is a question about
the build -- ``for_cpu`` is not registered for CUDA, the ``efficient`` kernel is
not registered for CPU -- so they are tried in turn and the answer remembered.
"""

ARGUMENTS = {
    "query": "q", "key": "k", "value": "v",
    "dropout_p": 0.0, "is_causal": False, "return_debug_mask": False,
    "compute_log_sumexp": True, "attn_bias": None, "attn_mask": None,
    "scale": "scale",
}
"""What to pass for each argument these ops are declared with.

By name and not by position: the arity differs between torch versions and
between the four ops -- ``flash`` took six arguments in one release and seven in
the next -- and a call built from the operator's own schema cannot be off by
one.  Anything not named here keeps the schema's default.
"""


def _call_by_schema(op, q, k, v, scale):
    """Call ``op`` with its arguments in whatever order it declares them.

    By name and by kind: some of these arguments are keyword-only in the
    operator's binding even though the schema lists them in one sequence, so
    passing everything positionally is an arity error on exactly the ops that
    would otherwise have worked.
    """

    supplied = {"q": q, "k": k, "v": v, "scale": scale}
    values, keywords = [], {}

    for argument in op._schema.arguments:
        if argument.name in ARGUMENTS:
            wanted = ARGUMENTS[argument.name]
            value = supplied.get(wanted, wanted) if isinstance(wanted, str) else wanted
        elif argument.has_default_value():
            value = argument.default_value
        else:
            raise TypeError(f"no value for {argument.name}")

        if argument.kwarg_only:
            keywords[argument.name] = value
        else:
            values.append(value)

    return op(*values, **keywords)


def _lse_attention(q, k, v, scale=None):
    """``(out, logsumexp)`` from whichever kernel on this build returns both.

    The merge needs the normaliser of the global attention, which the fast
    kernels all compute and only the private ops hand back.  Probed once and
    remembered: a build where none of them applies is not broken, it runs
    ``dense`` instead.
    """

    global _lse_kernel

    if _lse_kernel is False:
        return None
    if _lse_kernel is not None:
        return _shaped(_call_by_schema(_lse_kernel, q, k, v, scale), q)

    for name in LSE_OPS:
        op = getattr(torch.ops.aten, name, None)
        op = getattr(op, "default", None) if op is not None else None
        if op is None:
            continue
        try:
            result = _shaped(_call_by_schema(op, q, k, v, scale), q)
        except Exception:
            continue
        if result is not None:
            _lse_kernel = op
            return result

    _lse_kernel = False
    return None


_lse_kernel = None


def _shaped(result, q):
    """Check a private op gave back what its signature promises.

    Strict about it, because these are not public API: an op that started
    returning its outputs in another order would otherwise show up as an image
    that is subtly wrong rather than as a fallback to the dense path.
    """

    if not isinstance(result, (tuple, list)) or len(result) < 2:
        return None

    out, lse = result[0], result[1]
    if not isinstance(out, torch.Tensor) or not isinstance(lse, torch.Tensor):
        return None
    if out.shape != q.shape or lse.ndim != 3:
        return None
    if lse.shape[0] != q.shape[0] or lse.shape[1] != q.shape[1]:
        return None
    # some builds pad the log-sum-exp out to a multiple of the kernel's tile
    if lse.shape[2] < q.shape[2]:
        return None
    if not bool(torch.isfinite(lse[:, :, : q.shape[2]]).all()):
        return None

    return out, lse[:, :, : q.shape[2]].float()


def _region_attention(q, k, v, allowed, scale):
    """Attention over one region's keys, and its log-sum-exp per query.

    Written out rather than handed to a kernel because the key block is a
    handful of tokens: the scores are ``[B, H, L, n]`` with ``n`` in the tens,
    which is a rounding error beside the ``[B, H, L, D]`` the query already is,
    and doing it here is what makes ``allowed`` a vector instead of a matrix.
    """

    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.matmul(torch.softmax(scores, dim=-1), v.float())

    # a query outside the box has no weight on this region at all, which is a
    # normaliser of zero -- the merge below reads that as `e**-inf`
    lse = lse.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    return out, lse


def _merge(parts):
    """Combine ``(out, lse)`` pieces into the softmax over their union.

    The identity flash attention is built on, in fp32 and with the maximum
    pulled out so that a large score cannot overflow before it is normalised.
    """

    top = None
    for _, lse in parts:
        top = lse if top is None else torch.maximum(top, lse)
    top = torch.nan_to_num(top, neginf=0.0)

    total = None
    accumulated = None

    for out, lse in parts:
        weight = torch.exp(lse - top).unsqueeze(-1)
        contribution = out.float() * weight
        accumulated = contribution if accumulated is None else accumulated + contribution
        total = weight if total is None else total + weight

    return accumulated / total.clamp_min(torch.finfo(torch.float32).tiny)


def merged_attention(q, k, v, plan: Plan, total: int, scale=None, plain=None):
    """Regional attention without ever building an ``[L, L]`` mask.

    Three observations, in the order they save work:

    * The regions' keys are a contiguous tail of the text block, so the
      sequence splits in one place -- two slices and a concatenation, no
      gather.
    * A prompt in the batch with no regions of its own wants ordinary
      attention over every key, so it is not split at all: it goes to
      ``plain`` and costs exactly what it always cost.  This is the negative
      prompt of nearly every regional generation.
    * A query outside a box contributes nothing to that box's part of the
      merge, so the part is computed for the queries inside it and nowhere
      else.  The extra work is therefore proportional to the *area the boxes
      cover*, not to the picture, and not to how many boxes there are: eight
      small ones and one of the same total area cost the same.

    Returns ``None`` when this build cannot give back a log-sum-exp, or when
    the regions are not in the shape the split relies on, so the caller can
    fall back to :func:`dense_mask`.
    """

    spans = [span for row in plan.spans for span in row]
    if not spans:
        return None

    first = min(span.start for span in spans)
    last = max(span.stop for span in spans)
    if last > plan.geometry.txtlen or first >= last:
        # regions are appended to the prompt, so their tokens are the tail of
        # the text block; anything else is not this function's shape
        return None

    batch = q.shape[0]
    if len(plan.spans) not in (1, batch) and batch % len(plan.spans):
        return None

    per_item: list[list[Span]] = [
        plan.spans[index % len(plan.spans)] for index in range(batch)
    ]
    regioned = [index for index, row in enumerate(per_item) if row]
    if not regioned:
        return None

    rows = torch.tensor(regioned, device=q.device)
    globals_k = torch.cat((k[rows][:, :, :first], k[rows][:, :, last:]), dim=2)
    globals_v = torch.cat((v[rows][:, :, :first], v[rows][:, :, last:]), dim=2)

    base = _lse_attention(q[rows], globals_k, globals_v, scale)
    if base is None:
        return None

    base_out, base_lse = base
    out = torch.empty_like(q)

    untouched = [index for index in range(batch) if not per_item[index]]
    if untouched:
        others = torch.tensor(untouched, device=q.device)
        out[others] = _plain(q[others], k[others], v[others], scale, plain)

    for position, index in enumerate(regioned):
        out[index] = _merge_item(
            q, k, v, per_item[index], plan, total, scale,
            base_out[position], base_lse[position], index)

    return out


def _plain(q, k, v, scale, plain):
    """Ordinary attention, on the host's own backend where there is one."""

    if plain is not None:
        return plain(q, k, v, q.shape[1], mask=None, skip_reshape=True,
                     skip_output_reshape=True, scale=scale)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def _merge_item(q, k, v, spans: list[Span], plan: Plan, total: int, scale,
                base_out, base_lse, index: int):
    """One batch item's regions merged into its global attention.

    ``base_out`` is already right for every query the boxes do not reach: the
    global attention it came from had the regions' keys taken out of it, which
    is exactly what "cannot see this region" means.  So only the rows inside a
    box are recomputed, and they are gathered first so that everything after
    this line is the size of the boxes rather than the size of the picture.
    """

    allowed = [query_rows(span, plan.geometry, total, q.device) for span in spans]

    union = allowed[0]
    for other in allowed[1:]:
        union = union | other

    inside = union.nonzero().flatten()
    if inside.numel() == 0:
        return base_out

    queries = q[index : index + 1][:, :, inside]
    parts = [(base_out.unsqueeze(0)[:, :, inside], base_lse.unsqueeze(0)[:, :, inside])]

    for span, reach in zip(spans, allowed):
        parts.append(_region_attention(
            queries,
            k[index : index + 1][:, :, span.start : span.stop],
            v[index : index + 1][:, :, span.start : span.stop],
            reach[inside].unsqueeze(0),
            scale,
        ))

    merged = _merge(parts)[0].to(base_out.dtype)
    return base_out.clone().index_copy_(1, inside, merged)


def _shared_spans(per_item: list[list[Span]]) -> list[Span]:
    """Every distinct span in the batch, by position.

    Prompts in one batch are padded to a common length and the regions of the
    positive and the negative prompt need not line up, so a span is attended for
    the whole batch and switched off, per item, in :func:`_allowed_stack`.
    """

    seen: dict[tuple[int, int], Span] = {}
    for row in per_item:
        for span in row:
            seen.setdefault((span.start, span.stop), span)
    return [seen[key] for key in sorted(seen)]


def _allowed_stack(span: Span, per_item, plan: Plan, total: int, device):
    """``[B, L]`` of the queries each batch item lets this span reach.

    A batch item with no region at these positions is not a region switched
    off: prompts in a batch are padded to a common length, so the same indices
    are ordinary text in the negative prompt while they are a box in the
    positive one.  Ordinary text is visible to everything, so the row is all
    true -- and the split then adds back exactly the keys it took out, which is
    why an unregioned prompt in a regioned batch comes out bit-for-bit as it
    would have without this Extension.
    """

    rows = []
    for spans in per_item:
        mine = next(
            (s for s in spans if s.start == span.start and s.stop == span.stop),
            None,
        )
        if mine is None:
            rows.append(torch.ones(total, dtype=torch.bool, device=device))
        else:
            rows.append(query_rows(mine, plan.geometry, total, device))
    return torch.stack(rows, dim=0)


# ================================================================================ #
# The seam


_originals: dict[str, Callable] = {}


def installed() -> bool:
    return "attention_function" in _originals


def install(remove: bool = False) -> bool:
    """Replace ``attention_function`` in Krea 2's module with :func:`attend`.

    That name is the one seam this whole feature needs.  ``krea.py`` imports it
    at module level and calls it from every attention module in the model, and
    it is reached with the mask parameter that the blocks hard-code to ``None``
    -- so rebinding the name reaches every block, in the right place, without
    wrapping thirty modules or copying a forward that upstream may change.

    Everything that is not a regional generation goes straight through to the
    function that was there.
    """

    try:
        from backend.nn import krea
    except Exception:
        return False

    if remove:
        original = _originals.pop("attention_function", None)
        if original is not None:
            if getattr(krea.attention_function, "_negpip_regional", False):
                krea.attention_function = original
        return True

    if getattr(krea.attention_function, "_negpip_regional", False):
        return True

    _originals["attention_function"] = krea.attention_function
    krea.attention_function = attend
    return True


def _masked_backend(original: Callable) -> Callable:
    """A function that will take the dense mask without complaining.

    ``attention_flash`` asserts the mask is ``None`` and only then falls back to
    the masked path, logging the assertion as an error on the way -- once per
    block per step, which is some hundreds of error lines per image for a
    generation that is working perfectly.  Everything else in
    ``backend/attention.py`` accepts a mask, so only that one is stepped around.
    """

    try:
        from backend import attention
    except Exception:
        return original

    if original is getattr(attention, "attention_flash", None):
        return getattr(attention, "attention_pytorch", original)
    return original


def _reshaped(out: torch.Tensor, heads: int, skip_output_reshape: bool):
    """``[B, H, L, D]`` in the layout the caller expects back."""
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(out.shape[0], -1, heads * out.shape[-1])


def attend(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
           skip_output_reshape=False, **kwargs):
    """``attention_function``, with the regions of the running forward applied.

    Signature is the one every backend in ``backend/attention.py`` shares, and
    every path that is not a regional single-stream attention is handed to the
    function this one replaced -- including the text fusion blocks, whose
    sequence is the prompt alone and so cannot hold an image grid.
    """

    original = _originals.get("attention_function")
    if original is None:
        raise RuntimeError("NegPiP Regional lost the attention function it patched")

    plan = _plan
    if plan is None or mask is not None or not skip_reshape or q.ndim != 4:
        return original(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

    total = q.shape[2]
    if not plan.geometry.holds(total):
        return original(q, k, v, heads, mask=None, attn_precision=attn_precision,
                        skip_reshape=True, skip_output_reshape=skip_output_reshape,
                        **kwargs)

    scale = kwargs.get("scale", None)

    if plan.mode != "dense":
        merged = merged_attention(q, k, v, plan, total, scale, original)
        if merged is not None:
            _report(plan, "merge", total)
            return _reshaped(merged, heads, skip_output_reshape)
        if plan.mode == "merge":
            # asked for by name, so say why it is not what is running
            print("NegPiP Regional: no log-sum-exp from this attention build, "
                  "falling back to the dense mask")

    _report(plan, "dense", total)
    bias = dense_mask(plan, total, q.shape[0], q.device, q.dtype)
    return _masked_backend(original)(
        q, k, v, heads, mask=bias, attn_precision=attn_precision,
        skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)


_reported: bool = False
"""Whether this generation has already said the regions reached attention.

Module level and not on the plan: the plan is rebuilt on every forward, because
the patch grid it addresses is only known once the latent for that step is in
hand, so a flag living on it would say the line thirty times an image.
"""


def forget():
    """Let the next generation say its line again."""
    global _reported
    _reported = False


def _report(plan: Plan, how: str, total: int):
    """One line per generation saying the regions reached attention.

    The failure this fork is most likely to have is the quiet one -- a mask
    built for a geometry that is not the one being sampled, applied to nothing
    -- and an image that merely ignores a region looks exactly like an
    Extension that is not installed.
    """

    global _reported

    if _reported:
        return
    _reported = True

    counted = sum(len(row) for row in plan.spans)
    tokens = sum(span.length for row in plan.spans for span in row)
    geometry = plan.geometry
    print(
        f"NegPiP Regional Applied ({counted} region(s), {tokens} token(s), "
        f"{how}: {geometry.height}x{geometry.width} patches, "
        f"{total} in the stream)"
    )


attend._negpip_regional = True
"""The marker `install` recognises its own work by.

Without it a second install saves `attend` as the original it should fall
through to, and the first attention of the next generation recurses until the
stack ends -- which is why `test_installing_twice_does_not_wrap_twice` exists.
"""
