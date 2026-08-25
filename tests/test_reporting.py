"""The console lines, which are the only view of this from a running WebUI."""

import pytest

torch = pytest.importorskip("torch")


def _plan(regional):
    geometry = regional.Geometry(txtlen=12, height=6, width=5)
    spans = [[regional.Span(start=8, length=2, box=(0.0, 0.0, 1.0, 0.4))]]
    return regional.Plan(geometry=geometry, spans=spans)


def test_the_line_is_said_once_a_generation_not_once_a_step(regional, capsys):
    """The plan is rebuilt every forward; the flag must not live on it."""

    regional.forget()
    for _ in range(4):
        regional._report(_plan(regional), "merge", 42)

    said = capsys.readouterr().out.strip().splitlines()
    assert len(said) == 1
    assert "NegPiP Regional Applied" in said[0]


def test_the_next_generation_says_it_again(regional, capsys):
    regional.forget()
    regional._report(_plan(regional), "merge", 42)
    regional.forget()
    regional._report(_plan(regional), "merge", 42)
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_the_line_names_the_grid_it_was_built_for(regional, capsys):
    """A mask built for the wrong resolution is the quiet failure to catch."""

    regional.forget()
    regional._report(_plan(regional), "dense", 42)
    said = capsys.readouterr().out
    assert "6x5 patches" in said
    assert "1 region(s), 2 token(s), dense" in said
    assert "42 in the stream" in said
