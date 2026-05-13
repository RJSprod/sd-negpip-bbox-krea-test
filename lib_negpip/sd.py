from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from backend.nn.unet import CrossAttention
    from backend.nn.unet import IntegratedUNet2DConditionModel as UNet

import torch

from lib_negpip import IS_NEO
from modules import shared

if IS_NEO:
    from backend.attention import attention_function as optimized_attention
else:
    from ldm_patched.ldm.modules.attention import optimized_attention


def patch_sd_negpip(instance: "NegPiP", cls: "NegPiP", *, unpatch=False):
    if unpatch != cls._patched[0]:
        return

    cls._patched[0] = not cls._patched[0]

    unet: "UNet" = shared.sd_model.forge_objects.unet.model.diffusion_model
    for name, module in unet.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            _hook_forward(instance, module, unpatch)


# ================================================================================ #


class Counter:
    def __init__(self, xl: bool):
        self.count: int = 0
        self.limit: int = 70 if xl else 16
        self.p: bool = True

    def counter(self) -> bool:
        outpn = self.p

        self.count += 1
        if self.count == self.limit:
            self.p = not self.p
            self.count = 0

        return outpn


def _hook_forward(cls: "NegPiP", module: "CrossAttention", remove: bool):
    if remove:
        if hasattr(module, "orig_forward"):
            module.forward = module.orig_forward
            del module.orig_forward
        return

    counter = Counter(cls.is_xl)

    module.orig_forward = module.forward

    @torch.inference_mode()
    @wraps(module.orig_forward)
    def forward(x, context=None, value=None, mask=None, *args, **kwargs):

        @torch.inference_mode()
        def sub_forward(x, context, mask, conds, c_tokens, unconds, uc_tokens):
            if x.shape[0] == cls.batch_size * 2:
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

                xp = _main_forward(cls, module, ixp, contp, value, mask, c_tokens)
                xn = _main_forward(cls, module, ixn, contn, value, mask, uc_tokens)

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

                return _main_forward(cls, module, x, context, value, mask, tokens)

        return sub_forward(
            x,
            context,
            mask,
            cls.conds[0] if len(cls.conds) > 0 else None,
            cls.c_tokens[0] if len(cls.conds) > 0 else None,
            cls.unconds[0] if len(cls.unconds) > 0 else None,
            cls.uc_tokens[0] if len(cls.unconds) > 0 else None,
        )

    module.forward = forward


@torch.inference_mode()
def _main_forward(cls: "NegPiP", attn: "CrossAttention", x, ctx, value, mask, tokens):
    q = attn.to_q(x)
    ctx = ctx.to(x.dtype)
    k = attn.to_k(ctx)

    if value is not None:
        v = attn.to_v(value)
        del value
    else:
        v = attn.to_v(ctx)

    if cls.active:
        if tokens:
            v[:, -tokens:, :] = -v[:, -tokens:, :]

    out = optimized_attention(q, k, v, attn.heads, mask)
    return attn.to_out(out)
