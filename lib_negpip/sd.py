from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

import torch

from lib_negpip import IS_NEO
from modules import shared

if IS_NEO:
    from backend.attention import attention_function as optimized_attention
else:
    from ldm_patched.ldm.modules.attention import optimized_attention


class Counter:
    def __init__(self, xl: bool):
        self.count = 0
        self.limit = 70 if xl else 16
        self.p = True

    def counter(self):
        outpn = self.p

        self.count += 1
        if self.count == self.limit:
            self.p = not self.p
            self.count = 0

        return outpn


def hook_forwards(cls: "NegPiP", root_module: torch.nn.Module, *, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            _hook_forward(cls, module, remove)


def unload(cls: "NegPiP"):
    if hasattr(cls, "handle"):
        unet = shared.sd_model.forge_objects.unet.model.diffusion_model
        hook_forwards(cls, unet, remove=True)
        del cls.handle


def _hook_forward(cls: "NegPiP", module: torch.nn.Module, remove: bool):
    if remove:
        if hasattr(module, "orig_forward"):
            module.forward = module.orig_forward
            del module.orig_forward
        return

    counter = Counter(cls.is_xl)

    module.orig_forward = module.forward

    @torch.inference_mode()
    @wraps(module.orig_forward)
    def forward(
        x,
        context=None,
        value=None,
        mask=None,
        *args,
        **kwargs,
    ):

        @torch.inference_mode()
        def sub_forward(
            x,
            context,
            mask,
            conds,
            c_tokens,
            unconds,
            uc_tokens,
        ):
            if x.shape[0] == cls.batch * 2:
                if cls.rev:
                    contn, contp = context.chunk(2)
                    ixn, ixp = x.chunk(2)
                else:
                    contp, contn = context.chunk(2)
                    ixp, ixn = x.chunk(2)

                if conds is not None:
                    if contp.shape[0] != conds.shape[0]:
                        conds = conds.expand(contp.shape[0], -1, -1)
                    contp = torch.cat((contp, conds), 1)
                if unconds is not None:
                    if contn.shape[0] != unconds.shape[0]:
                        unconds = unconds.expand(contn.shape[0], -1, -1)
                    contn = torch.cat((contn, unconds), 1)

                xp = _main_forward(
                    cls,
                    module,
                    ixp,
                    contp,
                    value,
                    mask,
                    c_tokens,
                )
                xn = _main_forward(
                    cls,
                    module,
                    ixn,
                    contn,
                    value,
                    mask,
                    uc_tokens,
                )

                out = torch.cat([xn, xp]) if cls.rev else torch.cat([xp, xn])
                return out

            else:
                tokens = []
                concon = counter.count()
                if context.shape[1] == cls.c_len * 77 and concon:
                    if conds is not None:
                        if context.shape[0] != conds.shape[0]:
                            conds = conds.expand(context.shape[0], -1, -1)
                        context = torch.cat([context, conds], 1)
                        tokens = c_tokens
                elif context.shape[1] == cls.uc_len * 77 and concon:
                    if unconds is not None:
                        if context.shape[0] != unconds.shape[0]:
                            unconds = unconds.expand(context.shape[0], -1, -1)
                        context = torch.cat([context, unconds], 1)
                        tokens = uc_tokens

                return _main_forward(
                    cls,
                    module,
                    x,
                    context,
                    value,
                    mask,
                    tokens,
                )

        if (
            cls.conds is not None
            and cls.unconds is not None
            and len(cls.conds) > 0
            and len(cls.unconds) > 0
        ):
            return sub_forward(
                x,
                context,
                mask,
                cls.conds[0],
                cls.c_tokens[0],
                cls.unconds[0],
                cls.uc_tokens[0],
            )
        else:
            return sub_forward(
                x,
                context,
                mask,
                None,
                None,
                None,
                None,
            )

    module.forward = forward


@torch.inference_mode()
def _main_forward(
    cls: "NegPiP",
    attn,
    x,
    context,
    value=None,
    mask=None,
    tokens=[],
):
    q = attn.to_q(x)
    context = context.to(x.dtype)
    k = attn.to_k(context)
    if value is not None:
        v = attn.to_v(value)
        del value
    else:
        v = attn.to_v(context)

    if cls.active:
        if tokens:
            v[:, -tokens:, :] = -v[:, -tokens:, :]

    out = optimized_attention(q, k, v, attn.heads, mask)
    return attn.to_out(out)
