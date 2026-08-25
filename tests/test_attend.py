"""The seam: the one name this Extension rebinds inside Krea 2's module.

`backend.nn.krea` is the host's, so these tests stand a small one in its place
-- enough of it to be rebound, called the way the real blocks call it, and
restored.  What is asserted is the contract with the caller: same signature,
same output shape, and everything that is not a regional single-stream
attention handed straight back to the function that was there.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def plain_attention(q, k, v, heads, mask=None, attn_precision=None,
                    skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Stands in for `backend.attention.attention_pytorch`."""

    if not skip_reshape:
        b, _, dim = q.shape
        dim //= heads
        q, k, v = (t.view(b, -1, heads, dim).transpose(1, 2) for t in (q, k, v))

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(out.shape[0], -1, heads * out.shape[-1])


@pytest.fixture
def host(monkeypatch):
    """A `backend.nn.krea` and a `backend.attention` for `install` to find."""

    backend = types.ModuleType("backend")
    backend.__path__ = []
    nn = types.ModuleType("backend.nn")
    nn.__path__ = []
    krea = types.ModuleType("backend.nn.krea")
    krea.attention_function = plain_attention
    attention = types.ModuleType("backend.attention")
    attention.attention_pytorch = plain_attention
    attention.attention_flash = lambda *a, **k: pytest.fail("flash got a mask")
    nn.krea = krea
    backend.nn = nn
    backend.attention = attention

    for name, module in (
        ("backend", backend), ("backend.nn", nn),
        ("backend.nn.krea", krea), ("backend.attention", attention),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return krea


def _plan(regional, mode="auto", txtlen=12, height=6, width=5):
    geometry = regional.Geometry(txtlen=txtlen, height=height, width=width)
    spans = [[regional.Span(start=8, length=2, box=(0.0, 0.0, 1.0, 0.4))], []]
    return regional.Plan(geometry=geometry, spans=spans, mode=mode)


def _qkv(total, batch=2, heads=4, dim=16):
    torch.manual_seed(0)
    return tuple(torch.randn(batch, heads, total, dim) for _ in range(3))


def test_install_rebinds_the_name_and_puts_it_back(regional, host):
    assert host.attention_function is plain_attention
    assert regional.install(False)
    assert host.attention_function is regional.attend
    assert regional.install(True)
    assert host.attention_function is plain_attention


def test_installing_twice_does_not_wrap_twice(regional, host):
    regional.install(False)
    regional.install(False)
    regional.install(True)
    assert host.attention_function is plain_attention


def test_with_no_plan_the_call_goes_straight_through(regional, host):
    regional.install(False)
    regional.begin(None)
    try:
        q, k, v = _qkv(42)
        assert torch.equal(
            host.attention_function(q, k, v, 4, skip_reshape=True),
            plain_attention(q, k, v, 4, skip_reshape=True),
        )
    finally:
        regional.end()
        regional.install(True)


def test_the_text_fusion_blocks_are_left_alone(regional, host):
    """Their sequence is the prompt alone, which cannot hold an image grid."""

    regional.install(False)
    regional.begin(_plan(regional))
    try:
        q, k, v = _qkv(12)
        assert torch.equal(
            host.attention_function(q, k, v, 4, skip_reshape=True),
            plain_attention(q, k, v, 4, skip_reshape=True),
        )
    finally:
        regional.end()
        regional.install(True)


def test_a_regional_call_comes_back_in_the_callers_shape(regional, host):
    regional.install(False)
    plan = _plan(regional)
    regional.begin(plan)
    try:
        total = 12 + 30
        q, k, v = _qkv(total)
        out = host.attention_function(q, k, v, 4, skip_reshape=True)
        assert out.shape == (2, total, 4 * 16)
        assert out.dtype == q.dtype
    finally:
        regional.end()
        regional.install(True)


@pytest.mark.parametrize("mode", ["dense", "merge"])
def test_both_modes_produce_the_same_image(regional, host, mode):
    regional.install(False)
    total = 12 + 30
    q, k, v = _qkv(total)
    try:
        regional.begin(_plan(regional, mode=mode))
        got = host.attention_function(q, k, v, 4, skip_reshape=True)
        regional.end()

        regional.begin(_plan(regional, mode="dense"))
        reference = host.attention_function(q, k, v, 4, skip_reshape=True)
        regional.end()
    finally:
        regional.install(True)

    assert torch.allclose(got, reference, atol=1e-5, rtol=1e-4)


def test_a_mask_from_the_caller_is_respected_rather_than_replaced(regional, host):
    """Nothing passes one today, and a build that starts to must win."""

    regional.install(False)
    regional.begin(_plan(regional))
    try:
        total = 12 + 30
        q, k, v = _qkv(total)
        theirs = torch.zeros(2, 1, total, total)
        assert torch.equal(
            host.attention_function(q, k, v, 4, mask=theirs, skip_reshape=True),
            plain_attention(q, k, v, 4, mask=theirs, skip_reshape=True),
        )
    finally:
        regional.end()
        regional.install(True)


def test_the_dense_path_steps_around_the_flash_backend(regional, host):
    """`attention_flash` asserts the mask is None, and logs it as an error."""

    import backend.attention as attention

    host.attention_function = attention.attention_flash
    regional.install(False)
    regional.begin(_plan(regional, mode="dense"))
    try:
        total = 12 + 30
        q, k, v = _qkv(total)
        #  the fixture's flash fails the test if it is handed a mask
        out = host.attention_function(q, k, v, 4, skip_reshape=True)
        assert out.shape == (2, total, 4 * 16)
    finally:
        regional.end()
        regional.install(True)
