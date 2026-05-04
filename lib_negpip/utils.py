from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from modules.processing import StableDiffusionProcessing

import torch

from lib_negpip import IS_NEO

if IS_NEO:
    from backend.attention import attention_function as optimized_attention
else:
    from ldm_patched.ldm.modules.attention import optimized_attention


def hook_forwards(cls: "NegPiP", root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            if not remove:
                module.forward = __hook_forward(cls, module)
            else:
                del module.forward


def unload(cls: "NegPiP", p: "StableDiffusionProcessing"):
    if hasattr(cls, "handle"):
        unet = p.sd_model.forge_objects.unet.model.diffusion_model
        hook_forwards(cls, unet, remove=True)
        del cls.handle


def resetpcache(p: "StableDiffusionProcessing"):
    c = 3 if IS_NEO else 2
    p.cached_c = [None] * c
    p.cached_uc = [None] * c
    if hasattr(p, "cached_hr_c"):
        p.cached_hr_c = [None] * c
        p.cached_hr_uc = [None] * c


def hr_dealer(p: "StableDiffusionProcessing"):
    if not hasattr(p, "hr_prompts"):
        p.hr_prompts = None
    if not hasattr(p, "hr_negative_prompts"):
        p.hr_negative_prompts = None

    return bool(p.hr_prompts), bool(p.hr_negative_prompts)


def __hook_forward(cls: "NegPiP", module):
    @torch.inference_mode()
    def forward(
        x,
        context=None,
        mask=None,
        value=None,
        additional_tokens=None,
        *args,
        **kwargs,
    ):
        @torch.inference_mode()
        def sub_forward(
            x,
            context,
            mask,
            additional_tokens,
            conds,
            c_tokens,
            unconds,
            uc_tokens,
            latent=None,
        ):
            if x.shape[0] == cls.batch * 2:
                if cls.rev:
                    contn, contp = context.chunk(2)
                    ixn, ixp = x.chunk(2)
                else:
                    contp, contn = context.chunk(2)
                    ixp, ixn = x.chunk(2)  # x[0:cls.batch,:,:],x[cls.batch:,:,:]

                if conds is not None:
                    if contp.shape[0] != conds.shape[0]:
                        conds = conds.expand(contp.shape[0], -1, -1)
                    contp = torch.cat((contp, conds), 1)
                if unconds is not None:
                    if contn.shape[0] != unconds.shape[0]:
                        unconds = unconds.expand(contn.shape[0], -1, -1)
                    contn = torch.cat((contn, unconds), 1)

                xp = __main_forward(
                    cls,
                    module,
                    ixp,
                    contp,
                    value,
                    mask,
                    additional_tokens,
                    c_tokens,
                    args,
                    kwargs,
                )
                xn = __main_forward(
                    cls,
                    module,
                    ixn,
                    contn,
                    value,
                    mask,
                    additional_tokens,
                    uc_tokens,
                    args,
                    kwargs,
                )

                out = torch.cat([xn, xp]) if cls.rev else torch.cat([xp, xn])
                return out

            elif latent is not None:
                if latent:
                    conds = conds if conds is not None else None
                else:
                    conds = unconds if unconds is not None else None
                if conds is not None:
                    if context.shape[0] != conds.shape[0]:
                        conds = conds.expand(context.shape[0], -1, -1)
                    context = torch.cat([context, conds], 1)

                tokens = c_tokens if c_tokens is not None else uc_tokens

                return __main_forward(
                    cls,
                    module,
                    x,
                    context,
                    value,
                    mask,
                    additional_tokens,
                    tokens,
                    args,
                    kwargs,
                )

            else:
                tokens = []
                concon = __counter(cls.isxl)
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
                return __main_forward(
                    cls,
                    module,
                    x,
                    context,
                    value,
                    mask,
                    additional_tokens,
                    tokens,
                    args,
                    kwargs,
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
                additional_tokens,
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
                additional_tokens,
                None,
                None,
                None,
                None,
            )

    return forward


count = 0
p = True


def __counter(isxl: bool):
    global count, p
    count += 1

    limit = 70 if isxl else 16
    outpn = p

    if count == limit:
        p = not p
        count = 0
    return outpn


@torch.inference_mode()
def __main_forward(
    cls: "NegPiP",
    attn,
    x,
    context,
    value=None,
    mask=None,
    temb=None,
    tokens=[],
    args=None,
    kwargs=None,
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
