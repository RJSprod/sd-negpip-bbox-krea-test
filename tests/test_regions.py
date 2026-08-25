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


# ================================================================================ #
# A prompt does not arrive in the shape it was typed


def test_a_region_survives_its_line_break_being_lost(regions):
    """The failure that made the whole feature a no-op, silently.

    A prompt makes several hops between the box it was typed into and the text
    encoder, and at least one of them can flatten it.  The anchor was never
    what kept this from matching prose -- the four numbers are.
    """

    parsed = regions.split("a beach at sunset REGION 0 0 0.5 1 (person:-4)")
    assert parsed.scene == "a beach at sunset"
    assert parsed.regions[0].box == (0.0, 0.0, 0.5, 1.0)
    assert parsed.regions[0].text == "(person:-4)"


def test_a_flattened_prompt_with_two_regions_is_still_two(regions):
    parsed = regions.split(
        "a beach REGION 0 0 0.5 1 (person:-4) REGION 0.5 0 1 1 (dog:-2)")
    assert parsed.scene == "a beach"
    assert [r.text for r in parsed.regions] == ["(person:-4)", "(dog:-2)"]
    assert parsed.regions[1].box == (0.5, 0.0, 1.0, 1.0)


def test_a_region_after_a_comma_is_found(regions):
    parsed = regions.split("a beach, REGION 0 0 0.5 1 (person:-4)")
    assert parsed.scene == "a beach"
    assert len(parsed.regions) == 1


def test_the_word_still_needs_four_numbers_after_it(regions):
    """What keeps the relaxed match out of prose."""

    for said in ("a photograph of the REGION of Tuscany",
                 "REGION without numbers",
                 "the REGION 5 of the map"):
        assert regions.split(said).regions == []
        assert regions.split(said).scene == said


def test_a_flattened_region_runs_to_the_end_of_its_line(regions):
    """The one thing flattening costs, and it cannot be recovered.

    With the line break intact a region ends where the line does.  Without it
    there is nothing to say whether the words after the box were the region's
    or the scene's, so they are the region's -- which is why the syntax puts
    REGION lines at the end of a prompt.
    """

    parsed = regions.split("a beach, REGION 0 0 0.5 1 (person:-4), at sunset")
    assert parsed.scene == "a beach"
    assert parsed.regions[0].text == "(person:-4), at sunset"

    #  the line break settles it
    kept = regions.split("a beach, at sunset\nREGION 0 0 0.5 1 (person:-4)")
    assert kept.scene == "a beach, at sunset"
    assert kept.regions[0].text == "(person:-4)"


def test_lifting_a_region_out_of_a_line_leaves_no_debris(regions):
    parsed = regions.split(
        "a beach,  REGION 0 0 0.5 1 (person:-4)\n, at sunset")
    assert parsed.scene == "a beach, at sunset"


def test_a_prompt_that_is_only_a_region_leaves_an_empty_scene(regions):
    parsed = regions.split("REGION 0 0 1 1 everything")
    assert parsed.scene == ""
    assert parsed.regions[0].text == "everything"
