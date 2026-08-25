"""The diagnostic log: what it says, and that it says nothing when it is off."""

import importlib
import os

import pytest

torch = pytest.importorskip("torch")

PACKAGE = "lib_negpip_regional_tests"


@pytest.fixture
def probe(tmp_path, monkeypatch):
    module = importlib.import_module(f"{PACKAGE}.probe")
    monkeypatch.setattr(module, "PATH", str(tmp_path / "negpip_regional.log"))
    monkeypatch.setattr(module, "_opened", False)
    module._said.clear()
    module.enable(True)
    yield module
    module.enable(False)


def read(probe) -> str:
    return open(probe.PATH, encoding="utf-8").read() if os.path.exists(probe.PATH) else ""


def _plan(regional, boxes=((0.0, 0.0, 0.5, 1.0),), txtlen=12, height=6, width=5):
    spans = [[regional.Span(start=8 + 2 * i, length=2, box=box)
              for i, box in enumerate(boxes)], []]
    return regional.Plan(
        geometry=regional.Geometry(txtlen=txtlen, height=height, width=width),
        spans=spans)


# ================================================================================ #
# Off is off


def test_nothing_is_written_when_it_is_off(probe, regions):
    probe.enable(False)
    probe.begin("positive prompt")
    probe.prompt(regions.split("scene\nREGION 0 0 1 0.4 (man:-1)"))
    assert read(probe) == ""


def test_a_folder_it_cannot_write_to_is_not_an_error(probe, monkeypatch):
    monkeypatch.setattr(probe, "PATH", "/does/not/exist/anywhere/x.log")
    monkeypatch.setattr(probe, "_opened", False)
    probe.say("this goes to the console and nowhere else")


# ================================================================================ #
# What it records


def test_the_parsed_regions_are_recorded(probe, regions):
    probe.prompt(regions.split(
        "a landscape\nREGION 0 0 1 0.4 (man:-1)\nREGION 0 0.6 1 1 wildflowers"))
    said = read(probe)
    assert "2 region(s) parsed" in said
    assert "box (0.000, 0.000, 1.000, 0.400)" in said
    assert "'(man:-1)'" in said
    assert "'a landscape'" in said


def test_a_prompt_with_no_regions_says_so(probe, regions):
    probe.prompt(regions.split("just a landscape"))
    assert "no REGION lines" in read(probe)


def test_the_tokens_a_region_became_are_recorded(probe):
    boxes = [None] * 8 + [(0.0, 0.0, 1.0, 0.4)] * 3
    weights = [1.0] * 8 + [-4.0] * 3
    probe.tokens(boxes, weights, 11)
    said = read(probe)
    assert "region 1 -> 3 token(s) at 8..11 of 11" in said
    assert "-4.00" in said


def test_a_region_carrying_no_negative_weight_is_called_out(probe):
    """A box perfectly applied to a term with no sign is the quiet failure."""

    boxes = [None] * 4 + [(0.0, 0.0, 1.0, 0.4)] * 2
    probe.tokens(boxes, [1.0] * 6, 6)
    assert "no sign to apply inside it" in read(probe)


def test_the_conditioning_positions_are_recorded_once(probe, regional):
    plan = _plan(regional)
    for _ in range(5):
        probe.conditioning(plan.spans, 12)
    said = read(probe)
    assert said.count("conditioning: 2 prompt(s)") == 1
    assert "prompt 1: [8..10]" in said
    assert "prompt 2: no regions" in said


def test_a_new_generation_says_it_again(probe, regional):
    plan = _plan(regional)
    probe.conditioning(plan.spans, 12)
    probe.begin("positive prompt")
    probe.conditioning(plan.spans, 12)
    assert read(probe).count("conditioning: 2 prompt(s)") == 2


def test_the_geometry_is_drawn(probe, regional):
    """A shape on the screen is a thing somebody can see is wrong."""

    plan = _plan(regional, boxes=((0.0, 0.0, 0.5, 1.0),), height=16, width=16)
    probe.geometry(plan, 12 + 256)
    said = read(probe)
    assert "grid 16x16 = 256 patches" in said
    assert "covers 128 of 256 patches" in said
    #  the left half of the frame is the left half of the drawing
    drawn = [line.strip() for line in said.splitlines() if set(line.strip()) <= {"#", "."}]
    assert drawn and all(row.startswith("#") and row.endswith(".") for row in drawn)


def test_a_box_on_the_right_is_drawn_on_the_right(probe, regional):
    plan = _plan(regional, boxes=((0.5, 0.0, 1.0, 1.0),), height=16, width=16)
    probe.geometry(plan, 12 + 256)
    drawn = [line.strip() for line in read(probe).splitlines()
             if set(line.strip()) <= {"#", "."} and line.strip()]
    assert drawn and all(row.startswith(".") and row.endswith("#") for row in drawn)


# ================================================================================ #
# The measurement, which is the reason the module exists


def test_the_attention_share_is_measured_inside_and_outside(probe, regional):
    torch.manual_seed(0)
    plan = _plan(regional, height=6, width=5)
    total = 12 + 30
    q, k, v = (torch.randn(2, 4, total, 16) for _ in range(3))

    probe.attention(q, k, v, plan, total)
    said = read(probe)
    assert "attention on region 1:" in said
    assert "inside the box" in said and "outside the box" in said
    assert "% of each patch's attention" in said


def test_the_share_is_a_real_softmax_fraction(probe, regional):
    """Two keys out of forty-two, on random vectors, is a few per cent."""

    torch.manual_seed(0)
    plan = _plan(regional, height=6, width=5)
    total = 12 + 30
    q, k, v = (torch.randn(1, 4, total, 16) for _ in range(3))

    from importlib import import_module

    lse = import_module(f"{PACKAGE}.regional")._lse_attention
    share = probe._share(q, k, v, 0, torch.arange(12, 20), plan.spans[0][0], None, lse)
    assert 0.0 < share < 1.0


def test_a_measurement_that_cannot_be_made_is_not_an_error(probe, regional):
    plan = _plan(regional)
    #  tensors of the wrong shape entirely
    probe.attention(torch.zeros(1), torch.zeros(1), torch.zeros(1), plan, 42)
    assert "could not be measured" in read(probe) or read(probe)


# ================================================================================ #
# The gap that let a silent no-op stay silent


def test_the_prompt_is_recorded_even_when_nothing_parsed(probe, regions):
    """Logging only the successes is how five runs said nothing useful."""

    said_line = "a beach at sunset, two people"
    probe.prompt(regions.split(said_line), said_line)
    said = read(probe)
    assert "no REGION lines" in said
    assert "a beach at sunset, two people" in said


def test_a_newline_is_visible_in_the_record(probe, regions):
    """Whether the line breaks survived is the whole question."""

    said_line = "a beach\nREGION 0 0 0.5 1 (person:-4)"
    probe.prompt(regions.split(said_line), said_line)
    assert "\\n" in read(probe)


def test_a_very_long_prompt_is_clipped(probe, regions):
    said_line = "word, " * 400
    probe.prompt(regions.split(said_line), said_line)
    said = read(probe)
    assert "characters]" in said
    assert len(said) < 1200


def test_the_batch_end_of_the_bracket_is_recorded(probe):
    probe.batch(["a beach REGION 0 0 0.5 1 (person:-4)"], [""],
                "Krea2", True, True)
    said = read(probe)
    assert "model Krea2" in said
    assert "regions seen" in said
    assert "negative weights seen" in said
    assert "REGION 0 0 0.5 1" in said


def test_the_batch_record_says_when_it_sees_nothing(probe):
    probe.batch(["a beach"], [""], "Krea2", False, False)
    said = read(probe)
    assert "no regions seen" in said
    assert "no negative weights" in said
