"""The prompt syntax: what a REGION line means and what it leaves behind."""

import pytest


def test_a_prompt_without_the_keyword_is_returned_whole(regions):
    said = "a wide empty landscape at dawn, (man:-1)"
    parsed = regions.split(said)
    assert parsed.scene == said
    assert parsed.regions == []
    assert not parsed.regional


def test_a_region_line_leaves_the_scene_behind(regions):
    parsed = regions.split("a wide landscape\nREGION 0 0 1 0.4 (man:-1)")
    assert parsed.scene == "a wide landscape"
    assert len(parsed.regions) == 1
    assert parsed.regions[0].box == (0.0, 0.0, 1.0, 0.4)
    assert parsed.regions[0].text == "(man:-1)"


def test_commas_and_a_colon_are_allowed_between_the_numbers(regions):
    parsed = regions.split("scene\nREGION 0.5,0.5, 1, 1: (bird:2.0), flying")
    assert parsed.regions[0].box == (0.5, 0.5, 1.0, 1.0)
    assert parsed.regions[0].text == "(bird:2.0), flying"


def test_the_keyword_in_running_text_is_left_alone(regions):
    said = "a photograph of the REGION of Tuscany"
    assert regions.split(said).scene == said
    assert not regions.split(said).regional


def test_corners_may_be_given_in_either_order(regions):
    parsed = regions.split("scene\nREGION 1 1 0 0 something")
    assert parsed.regions[0].box == (0.0, 0.0, 1.0, 1.0)


def test_coordinates_outside_the_frame_mean_the_edge(regions):
    parsed = regions.split("scene\nREGION -0.5 0 2 0.5 something")
    assert parsed.regions[0].box == (0.0, 0.0, 1.0, 0.5)


def test_a_box_with_no_area_is_dropped(regions):
    parsed = regions.split("scene\nREGION 0.5 0.5 0.5 0.9 nothing")
    assert parsed.regions == []
    assert parsed.scene == "scene"


def test_a_region_with_no_text_is_dropped(regions):
    parsed = regions.split("scene\nREGION 0 0 1 1   ")
    assert parsed.regions == []


def test_the_combined_prompt_is_the_scene_then_the_regions_in_order(regions):
    parsed = regions.split(
        "scene\nREGION 0 0 1 0.4 sky things\nREGION 0 0.6 1 1 ground things")
    assert parsed.combined == "scene, sky things, ground things"


def test_regions_may_carry_no_weight_at_all(regions):
    parsed = regions.split("scene\nREGION 0 0 1 0.4 a flock of birds")
    assert parsed.regions[0].text == "a flock of birds"


def test_token_spans_follow_the_scene_in_order(regions):
    assert regions.token_span(12, [4, 6]) == [(12, 4), (16, 6)]
    assert regions.token_span(0, []) == []


@pytest.mark.parametrize(
    "box, expected",
    [
        ((0.0, 0.0, 1.0, 1.0), (0, 0, 10, 10)),
        ((0.0, 0.0, 1.0, 0.5), (0, 0, 5, 10)),
        ((0.5, 0.5, 1.0, 1.0), (5, 5, 10, 10)),
    ],
)
def test_a_box_becomes_a_rectangle_of_patches(regions, box, expected):
    assert regions.patch_bounds(box, 10, 10) == expected


def test_a_thin_box_still_covers_a_patch(regions):
    # rounded outward, so a sliver is the patches it touches and never nothing
    top, left, bottom, right = regions.patch_bounds((0.01, 0.01, 0.02, 0.02), 10, 10)
    assert bottom > top and right > left


def test_patch_indices_are_row_major(regions):
    found = regions.patch_indices((0.5, 0.0, 1.0, 0.5), 4, 4)
    #  columns 2..3 of rows 0..1, in the order the latent is flattened
    assert found == [2, 3, 6, 7]


def test_flattening_keeps_the_terms_and_drops_the_boxes(regions):
    """For every model that is not Krea 2; see `utils.flatten`."""

    import importlib

    utils = importlib.import_module(regions.__name__.rsplit(".", 1)[0] + ".utils")
    said = "a landscape\nREGION 0 0 1 0.4 (man:-1)\nREGION 0 0.6 1 1 wildflowers"
    assert utils.flatten(said) == "a landscape, (man:-1), wildflowers"


def test_flattening_leaves_a_plain_prompt_exactly_as_it_was(regions):
    import importlib

    utils = importlib.import_module(regions.__name__.rsplit(".", 1)[0] + ".utils")
    said = "a landscape,  with   odd   spacing "
    assert utils.flatten(said) == said
