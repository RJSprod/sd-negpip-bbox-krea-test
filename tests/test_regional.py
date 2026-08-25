"""The mask itself: what it blocks, and that the fast path agrees with it.

The dense mask is the definition -- fifteen lines that say plainly which query
may see which key -- and the merge is an optimisation of it.  So most of what
is worth asserting is either "the definition says what we meant" or "the fast
one still says the same thing", and the second is the one that would break
quietly.
"""

import math

import pytest

torch = pytest.importorskip("torch")


def _plan(regional, spans, txtlen=12, height=6, width=5):
    geometry = regional.Geometry(txtlen=txtlen, height=height, width=width)
    return regional.Plan(geometry=geometry, spans=spans)


def _span(regional, start, length, box):
    return regional.Span(start=start, length=length, box=box)


def _qkv(total, batch=2, heads=4, dim=16, dtype=torch.float32):
    torch.manual_seed(0)
    return tuple(torch.randn(batch, heads, total, dim, dtype=dtype) for _ in range(3))


# ================================================================================ #
# Geometry


def test_the_text_fusion_sequence_is_not_a_combined_one(regional):
    geometry = regional.Geometry(txtlen=12, height=6, width=5)
    #  the fusion blocks attend over the prompt alone
    assert not geometry.holds(12)
    #  and the single-stream blocks over text, references and image
    assert geometry.holds(12 + 30)
    assert geometry.reflen(12 + 3 + 30) == 3


def test_a_forward_with_no_image_grid_is_not_ours(regional):
    assert not regional.Geometry(txtlen=12, height=0, width=0).holds(99)


def test_only_image_patches_inside_the_box_may_look(regional):
    plan = _plan(regional, [[_span(regional, 8, 2, (0.0, 0.0, 1.0, 0.5))]])
    total = 12 + 30
    allowed = regional.query_rows(plan.spans[0][0], plan.geometry, total, "cpu")

    #  the scene's own tokens are shut out, deliberately: see `query_rows`
    assert not bool(allowed[:8].any())
    #  the region's tokens read the prompt they are part of
    assert bool(allowed[8:10].all())
    #  the top half of a 6x5 grid is the first three rows
    image = allowed[12:]
    assert bool(image[: 3 * 5].all())
    assert not bool(image[3 * 5 :].any())


# ================================================================================ #
# The dense mask


def test_the_dense_mask_blocks_a_region_from_everything_outside_it(regional):
    plan = _plan(regional, [[_span(regional, 8, 2, (0.0, 0.0, 1.0, 0.5))]])
    total = 12 + 30
    bias = regional.dense_mask(plan, total, 1, "cpu", torch.float32)

    blocked = torch.finfo(torch.float32).min
    #  a patch in the bottom half cannot see the region's keys
    assert float(bias[0, 0, total - 1, 8]) == blocked
    #  a patch in the top half can
    assert float(bias[0, 0, 12, 8]) == 0.0
    #  and nothing else in the mask is touched at all
    assert float(bias[0, 0, :, :8].abs().max()) == 0.0
    assert float(bias[0, 0, :, 10:].abs().max()) == 0.0


def test_a_batch_item_with_no_regions_is_not_masked(regional):
    plan = _plan(regional, [[_span(regional, 8, 2, (0.0, 0.0, 1.0, 0.5))], []])
    bias = regional.dense_mask(plan, 12 + 30, 2, "cpu", torch.float32)
    assert float(bias[1].abs().max()) == 0.0


def test_the_blocked_value_is_finite(regional):
    #  an -inf that meets a fused kernel's padding is a NaN for the whole row,
    #  and a NaN at step three of thirty is a black image with no error on it
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        assert math.isfinite(regional._neutral(dtype))


# ================================================================================ #
# The merge


def test_the_merge_agrees_with_the_dense_mask(regional):
    plan = _plan(regional, [
        [_span(regional, 8, 2, (0.0, 0.0, 1.0, 0.4)),
         _span(regional, 10, 2, (0.5, 0.5, 1.0, 1.0))],
        [],
    ])
    total = 12 + 3 + 30
    q, k, v = _qkv(total)

    bias = regional.dense_mask(plan, total, q.shape[0], q.device, q.dtype)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    merged = regional.merged_attention(q, k, v, plan, total)
    if merged is None:
        pytest.skip("no log-sum-exp from this attention build")

    assert torch.allclose(merged, reference, atol=1e-5, rtol=1e-4)


def test_the_merge_agrees_when_both_prompts_have_regions(regional):
    plan = _plan(regional, [
        [_span(regional, 9, 3, (0.0, 0.0, 0.5, 1.0))],
        [_span(regional, 9, 3, (0.5, 0.0, 1.0, 1.0))],
    ])
    total = 12 + 30
    q, k, v = _qkv(total)

    bias = regional.dense_mask(plan, total, q.shape[0], q.device, q.dtype)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    merged = regional.merged_attention(q, k, v, plan, total)
    if merged is None:
        pytest.skip("no log-sum-exp from this attention build")

    assert torch.allclose(merged, reference, atol=1e-5, rtol=1e-4)


def test_an_unregioned_prompt_in_a_regioned_batch_is_untouched(regional):
    """The split takes keys out and puts them back; it must put back all of them."""

    plan = _plan(regional, [[_span(regional, 8, 2, (0.0, 0.0, 1.0, 0.4))], []])
    total = 12 + 30
    q, k, v = _qkv(total)

    merged = regional.merged_attention(q, k, v, plan, total)
    if merged is None:
        pytest.skip("no log-sum-exp from this attention build")

    plain = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(merged[1], plain[1], atol=1e-5, rtol=1e-4)


def test_the_merge_declines_a_shape_it_cannot_split(regional):
    #  a region that is not in the text block is not the contiguous tail the
    #  split relies on, and declining is what makes the dense path the fallback
    plan = _plan(regional, [[_span(regional, 20, 2, (0.0, 0.0, 1.0, 0.4))]])
    total = 12 + 30
    q, k, v = _qkv(total)
    assert regional.merged_attention(q, k, v, plan, total) is None


def test_the_merge_declines_when_there_are_no_regions(regional):
    plan = _plan(regional, [[], []])
    q, k, v = _qkv(42)
    assert regional.merged_attention(q, k, v, plan, 42) is None


# ================================================================================ #
# The table the plan travels in


def test_a_table_round_trips_through_the_conditioning_shape(regional, regions):
    parsed = regions.split(
        "scene\nREGION 0 0 1 0.4 sky\nREGION 0 0.6 1 1 ground")
    table = regional.table_from_regions(parsed.regions, [2, 3], 8, 16)
    assert table.shape == (16, regional.COLUMNS)

    spans = regional.spans_from_table(table.unsqueeze(0), 16)[0]
    assert [(s.start, s.length) for s in spans] == [(8, 2), (10, 3)]
    assert spans[0].box == (0.0, 0.0, 1.0, 0.4)
    assert spans[1].box == (0.0, 0.6, 1.0, 1.0)


def test_padding_a_prompt_adds_no_regions(regional):
    table = torch.zeros(2, 16, regional.COLUMNS)
    table[0, 4:6, 0] = 1.0
    table[0, 4:6, 1:] = torch.tensor([0.0, 0.0, 1.0, 0.5])
    spans = regional.spans_from_table(table, 16)
    assert [(s.start, s.length) for s in spans[0]] == [(4, 2)]
    assert spans[1] == []


def test_two_adjacent_regions_with_the_same_box_are_one(regional):
    table = torch.zeros(1, 8, regional.COLUMNS)
    table[0, 2:6, 0] = 1.0
    table[0, 2:6, 1:] = torch.tensor([0.0, 0.0, 1.0, 0.5])
    spans = regional.spans_from_table(table, 8)[0]
    assert [(s.start, s.length) for s in spans] == [(2, 4)]


def test_two_adjacent_regions_with_different_boxes_are_two(regional):
    table = torch.zeros(1, 8, regional.COLUMNS)
    table[0, 2:4, 0] = 1.0
    table[0, 2:4, 1:] = torch.tensor([0.0, 0.0, 1.0, 0.5])
    table[0, 4:6, 0] = 1.0
    table[0, 4:6, 1:] = torch.tensor([0.0, 0.5, 1.0, 1.0])
    spans = regional.spans_from_table(table, 8)[0]
    assert [(s.start, s.length) for s in spans] == [(2, 2), (4, 2)]


def test_a_table_that_is_not_one_is_ignored(regional):
    assert regional.spans_from_table(None, 8) == []
    assert regional.spans_from_table(torch.zeros(1, 8, 3), 8) == []
    assert regional.spans_from_table(torch.zeros(8), 8) == []


def test_only_the_text_block_is_read(regional):
    #  the table is as long as the conditioning; anything past the prompt is
    #  padding from a longer prompt in the same batch
    table = torch.zeros(1, 16, regional.COLUMNS)
    table[0, 12:14, 0] = 1.0
    table[0, 12:14, 1:] = torch.tensor([0.0, 0.0, 1.0, 0.5])
    assert regional.spans_from_table(table, 12)[0] == []


# ================================================================================ #
# Text fusion, which runs before any of the above


def test_the_fusion_mask_stops_the_scene_reading_a_region(regional):
    """The leak that made a regional negative perturb everything and negate nothing."""

    plan = _plan(regional, [[_span(regional, 8, 2, (0.0, 0.0, 1.0, 0.4))]])
    bias = regional.fusion_mask(plan, 12, 1, "cpu", torch.float32)
    blocked = torch.finfo(torch.float32).min

    #  no scene token may read the region's tokens
    assert float(bias[0, 0, 0, 8]) == blocked
    assert float(bias[0, 0, 7, 9]) == blocked
    #  the region reads itself, so the fragment is still encoded as a phrase
    assert float(bias[0, 0, 8, 9]) == 0.0
    #  and it still reads the scene, so it knows what picture it is in
    assert float(bias[0, 0, 8, 0]) == 0.0
    #  nothing else is touched
    assert float(bias[0, 0, :, :8].abs().max()) == 0.0


def test_two_regions_do_not_read_each_other_in_fusion(regional):
    plan = _plan(regional, [[
        _span(regional, 8, 2, (0.0, 0.0, 1.0, 0.4)),
        _span(regional, 10, 2, (0.5, 0.5, 1.0, 1.0)),
    ]])
    bias = regional.fusion_mask(plan, 12, 1, "cpu", torch.float32)
    blocked = torch.finfo(torch.float32).min

    assert float(bias[0, 0, 8, 10]) == blocked
    assert float(bias[0, 0, 10, 8]) == blocked
    assert float(bias[0, 0, 10, 11]) == 0.0


def test_the_refiner_blocks_are_recognised_and_the_layerwise_ones_are_not(regional):
    """Both reach the patched function with a sequence that is not the stream."""

    #  one row of the region table per prompt, so the plan knows the batch
    plan = _plan(regional, [[_span(regional, 8, 2, (0.0, 0.0, 1.0, 0.4))], []])

    #  the refiner attends over the tokens, with the batch as itself
    assert regional._is_fusion(plan, 12, 2)
    #  the layerwise stack folds the tokens into the batch and attends over the
    #  tapped layers, where a token index would mean something else entirely
    assert not regional._is_fusion(plan, 4, 2 * 12)
    assert not regional._is_fusion(plan, 12, 2 * 12)
    #  and the combined stream is not fusion at all
    assert not regional._is_fusion(plan, 12 + 30, 2)


def test_a_plan_with_no_prompts_in_it_masks_no_fusion(regional):
    plan = _plan(regional, [])
    assert plan.prompts == 0
    assert not regional._is_fusion(plan, 12, 2)
