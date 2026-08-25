"""The path a region takes from the prompt to the attention matrix.

The real one runs through Forge's text engine, its conditioning machinery and
Krea 2's transformer, none of which are here.  What can be tested without them
is the arithmetic between the ends: a prompt goes in, a table comes out with
the regions on the right tokens, and the plan built from that table addresses
the patches the box was drawn over.
"""

import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

PACKAGE = "lib_negpip_regional_tests"


@pytest.fixture
def krea2(monkeypatch, regional):
    """Import `krea2` with the host modules it reaches for stood in."""

    def module(name, **attributes):
        made = types.ModuleType(name)
        made.__path__ = []
        for key, value in attributes.items():
            setattr(made, key, value)
        monkeypatch.setitem(sys.modules, name, made)
        return made

    module("backend")
    module("backend.args", dynamic_args={})
    module("backend.sampling")
    module("backend.sampling.condition", compile_conditions=lambda c: c,
           Condition=object, ConditionCrossAttn=object)
    module("backend.sampling.sampling_function", compile_conditions=lambda c: c)
    module("backend.nn")
    module("backend.nn.krea", attention_function=lambda *a, **k: None)
    module("backend.attention")
    module("modules")
    module("modules.shared", opts=types.SimpleNamespace(emphasis="Original"))

    return importlib.import_module(f"{PACKAGE}.krea2")


def test_the_table_puts_each_region_on_its_own_tokens(krea2, regional, regions):
    parsed = regions.split(
        "a landscape\nREGION 0 0 1 0.4 (man:-1)\nREGION 0.5 0.5 1 1 (bird:2)")

    #  the scene is eight tokens, then two for the first region, three for the
    #  second -- which is what the engine's tokenizer would have produced
    boxes = [None] * 8
    boxes += [parsed.regions[0].box] * 2
    boxes += [parsed.regions[1].box] * 3

    table = krea2._region_table(boxes, torch.zeros(13, 4, 8))
    assert table.shape == (13, regional.COLUMNS)

    spans = regional.spans_from_table(table.unsqueeze(0), 13)[0]
    assert [(s.start, s.length) for s in spans] == [(8, 2), (10, 3)]
    assert spans[0].box == (0.0, 0.0, 1.0, 0.4)


def test_the_stripped_template_is_taken_off_the_front(krea2, regional):
    """`strip_template` removes a prefix and keeps the tail, so the tail lines up."""

    boxes = [None] * 30 + [(0.0, 0.0, 1.0, 0.5)] * 2
    table = krea2._region_table(boxes, torch.zeros(4, 4, 8))
    #  four tokens survived the strip, and the last two of the boxes are theirs
    assert [float(v) for v in table[:, 0]] == [0.0, 0.0, 1.0, 1.0]


def test_a_prompt_with_no_regions_makes_an_empty_table(krea2, regional):
    table = krea2._region_table([None] * 6, torch.zeros(6, 4, 8))
    assert float(table.abs().max()) == 0.0
    assert regional.spans_from_table(table.unsqueeze(0), 6)[0] == []


def test_the_weights_and_the_regions_are_expanded_together(krea2):
    """An image becomes several embeddings; both lists have to follow it."""

    tokens = [1, {"type": "image"}, 2]
    inserts = [{"index": 1, "size": 3}]

    weights = krea2._align(tokens, [1.0, 1.0, -2.0], inserts, 5, 1.0)
    boxes = krea2._align(tokens, [None, None, (0.0, 0.0, 1.0, 0.5)], inserts, 5, None)

    assert weights == [1.0, 1.0, 1.0, 1.0, -2.0]
    assert boxes == [None, None, None, None, (0.0, 0.0, 1.0, 0.5)]


def test_alignment_that_does_not_add_up_is_refused(krea2):
    #  a silent misalignment puts a box on the wrong tokens, which reads as the
    #  coordinates being wrong rather than as a bug
    assert krea2._align([1, 2], [1.0], [], 2, 1.0) is None
    assert krea2._align([1, 2], [1.0, 1.0], [], 5, 1.0) is None


def test_the_plan_is_built_from_the_latent_being_sampled(krea2, regional):
    """Highres fix samples two resolutions from one prompt; the grid follows."""

    table = torch.zeros(1, 10, regional.COLUMNS)
    table[0, 8:10, 0] = 1.0
    table[0, 8:10, 1:] = torch.tensor([0.0, 0.0, 1.0, 0.5])

    dit = types.SimpleNamespace(patch=2)
    context = torch.zeros(1, 10, 8)

    small = krea2._plan(dit, (torch.zeros(1, 4, 1, 64, 64), None, context), {}, table)
    large = krea2._plan(dit, (torch.zeros(1, 4, 1, 128, 96), None, context), {}, table)

    assert (small.geometry.height, small.geometry.width) == (32, 32)
    assert (large.geometry.height, large.geometry.width) == (64, 48)
    assert small.geometry.txtlen == 10
    assert [(s.start, s.length) for s in small.spans[0]] == [(8, 2)]


def test_a_latent_that_does_not_divide_by_the_patch_size_rounds_up(krea2):
    #  the model pads up to the patch size before it makes its grid
    dit = types.SimpleNamespace(patch=2)
    table = torch.zeros(1, 4, 5)
    table[0, 2:4, 0] = 1.0
    table[0, 2:4, 3:] = 1.0
    plan = krea2._plan(dit, (torch.zeros(1, 4, 1, 63, 65), None, torch.zeros(1, 4, 8)),
                       {}, table)
    assert (plan.geometry.height, plan.geometry.width) == (32, 33)


def test_no_table_is_no_plan(krea2):
    dit = types.SimpleNamespace(patch=2)
    assert krea2._plan(dit, (torch.zeros(1, 4, 1, 64, 64), None, torch.zeros(1, 4, 8)),
                       {}, None) is None


def test_a_table_with_no_regions_in_it_is_no_plan(krea2, regional):
    dit = types.SimpleNamespace(patch=2)
    table = torch.zeros(1, 8, regional.COLUMNS)
    assert krea2._plan(dit, (torch.zeros(1, 4, 1, 64, 64), None, torch.zeros(1, 8, 8)),
                       {}, table) is None


# ================================================================================ #
# A region owns its words, not the punctuation around them


def test_punctuation_is_not_part_of_a_region(krea2):
    assert krea2._words("person")
    assert krea2._words("a tall red lighthouse")
    assert krea2._words("2 people")
    assert not krea2._words(", ")
    assert not krea2._words(",")
    assert not krea2._words("  ")
    assert not krea2._words("")
    assert not krea2._words(None)


def test_the_region_table_skips_the_separator(krea2, regional, regions):
    """Four tokens confined and one signed is three quarters of a boost wasted."""

    parsed = regions.split("a beach, REGION 0 0 0.5 1 (person:-4),")
    #  what the fragments come out as: separator, the word, the trailing comma
    boxes = [None] * 6 + [None, parsed.regions[0].box, None]

    table = krea2._region_table(boxes, torch.zeros(9, 4, 8))
    spans = regional.spans_from_table(table.unsqueeze(0), 9)[0]

    assert [(s.start, s.length) for s in spans] == [(7, 1)]
