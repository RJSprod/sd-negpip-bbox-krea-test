import re

import torch
from ldm_patched.ldm.modules.attention import default, optimized_attention
from modules import scripts
from modules.prompt_parser import (
    get_learned_conditioning,
    get_learned_conditioning_prompt_schedules,
)
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser

neg_pattern = r"\([^\(\:\)]+\:\s*\-\d+(?:\.[\d]+)?\s*\)"


class NegPiP(scripts.Script):
    def __init__(self):
        self.active = True

        self.conds = None
        self.c_len = []
        self.c_tokens = []

        self.unconds = None
        self.uc_len = []
        self.uc_tokens = []

        self.hr = False
        self.x = None

    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def process_batch(self, p, *args, **kwargs):
        self.__init__()
        flag = False

        self.hrp, self.hrn = hr_dealer(p)

        self.batch = p.batch_size
        self.isxl = p.sd_model.is_sdxl
        self.rev = p.sampler_name not in ["DDIM", "PLMS", "UniPC"]

        tokenizer = (
            p.sd_model.conditioner.embedders[0].tokenize_line
            if self.isxl
            else p.sd_model.cond_stage_model.tokenize_line
        )

        def getScheduledNegs(scheduled, prompts):
            output = []
            nonlocal flag

            for i, batch_schedule in enumerate(scheduled):
                stepout = []
                seps = None

                for step, prompt in batch_schedule:
                    sep_prompts = prompt.split(seps) if seps else [prompt]
                    pad = 0
                    padtextweight = []
                    for sep_prompt in sep_prompts:
                        neg_matches = re.findall(neg_pattern, sep_prompt)
                        neg_targets = []
                        text_weights = []
                        for minusmatch in neg_matches:
                            neg_targets.append(minusmatch.strip("(").strip(")"))
                            prompts[i] = prompts[i].replace(minusmatch, "")
                        neg_targets = [x.split(":") for x in neg_targets]
                        for text, weight in neg_targets:
                            if text.strip() in ("BREAK", "AND"):
                                continue
                            weight = float(weight)
                            if weight < 0.0:
                                text_weights.append([text, weight])
                                flag = True
                        padtextweight.append([pad, text_weights])
                        tokens, tokensnum = tokenizer(sep_prompt)
                        pad = tokensnum // 75 + 1 + pad
                    stepout.append([step, padtextweight])
                output.append(stepout)
            return output

        scheduled_p = get_learned_conditioning_prompt_schedules(p.prompts, p.steps)
        scheduled_np = get_learned_conditioning_prompt_schedules(
            p.negative_prompts, p.steps
        )

        if self.hrp:
            scheduled_hr_p = get_learned_conditioning_prompt_schedules(
                p.hr_prompts,
                p.hr_second_pass_steps if p.hr_second_pass_steps > 0 else p.steps,
            )
        if self.hrn:
            scheduled_hr_np = get_learned_conditioning_prompt_schedules(
                p.hr_negative_prompts,
                p.hr_second_pass_steps if p.hr_second_pass_steps > 0 else p.steps,
            )

        nip = getScheduledNegs(scheduled_p, p.prompts)
        pin = getScheduledNegs(scheduled_np, p.negative_prompts)

        if self.hrp:
            hr_nip = getScheduledNegs(scheduled_hr_p, p.hr_prompts)
        if self.hrn:
            hr_pin = getScheduledNegs(scheduled_hr_np, p.hr_negative_prompts)

        if not flag:
            self.active = False
            unload(self, p)
            return

        def cond_dealer(targets):
            conds = []
            start = None
            end = None
            for target in targets:
                input = SdConditioning(
                    [f"({target[0]}:{-target[1]})"], width=p.width, height=p.height
                )
                cond = get_learned_conditioning(p.sd_model, input, p.steps)
                if start is None:
                    start = (
                        cond[0][0].cond[0:1, :]
                        if not self.isxl
                        else cond[0][0].cond["crossattn"][0:1, :]
                    )
                if end is None:
                    end = (
                        cond[0][0].cond[-1:, :]
                        if not self.isxl
                        else cond[0][0].cond["crossattn"][-1:, :]
                    )
                token, tokenlen = tokenizer(target[0])
                conds.append(
                    cond[0][0].cond[1 : tokenlen + 2, :]
                    if not self.isxl
                    else cond[0][0].cond["crossattn"][1 : tokenlen + 2, :]
                )
            conds = torch.cat(conds, 0)
            conds = conds.unsqueeze(0)
            return conds.repeat(self.batch, 1, 1), conds.shape[1]

        def calc_conds(targetlist):
            outconds = []
            for batch in targetlist:
                stepconds = []
                for step, regions in batch:
                    regionconds = []
                    for region, targets in regions:
                        if targets:
                            conds, c_tokens = cond_dealer(targets)
                            regionconds.append([region, conds, c_tokens])
                        else:
                            regionconds.append([region, None, None])
                    stepconds.append([step, regionconds])
                outconds.append(stepconds)
            return outconds

        self.conds_all = calc_conds(nip)
        self.unconds_all = calc_conds(pin)

        if self.hrp:
            self.hr_conds_all = calc_conds(hr_nip)
        if self.hrn:
            self.hr_unconds_all = calc_conds(hr_pin)

        resetpcache(p)

        def calcSets(A, B):
            return A // B if A % B == 0 else A // B + 1

        self.c_len = calcSets(tokenizer(p.prompts[0])[1], 75)
        self.uc_len = calcSets(tokenizer(p.negative_prompts[0])[1], 75)

        if not hasattr(self, "negpip_dr_callbacks"):
            self.negpip_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

        self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)

        print(
            "\n".join(
                [
                    "",
                    "NegPiP Enable",
                    f" - Positive: {self.conds_all[0][0][1][0][2]}",
                    f" - Negative: {self.unconds_all[0][0][1][0][2]}",
                    "",
                ]
            )
        )

        p.extra_generation_params["NegPiP"] = True

    def postprocess(self, p, processed, *args):
        unload(self, p)
        self.conds_all = None
        self.unconds_all = None

    def denoiser_callback(self, params: CFGDenoiserParams):
        if not self.active:
            return

        if self.x is None:
            self.x = params.x.shape
        if self.x != params.x.shape:
            self.hr = True

        conds_list = []
        tokens_list = []

        conds = self.hr_conds_all if self.hr and self.hrp else self.conds_all
        if conds is not None:
            for step, regions in conds[0]:
                if step >= params.sampling_step + 2:
                    for region, conds, tokens in regions:
                        conds_list.append(conds)
                        tokens_list.append(tokens)
                    break
            self.conds = conds_list
            self.c_tokens = tokens_list

        unconds_list = []
        uc_tokens_list = []

        unconds = self.hr_unconds_all if self.hr and self.hrn else self.unconds_all
        if unconds is not None:
            for step, regions in unconds[0]:
                if step >= params.sampling_step + 2:
                    for region, unconds, uc_tokens in regions:
                        unconds_list.append(unconds)
                        uc_tokens_list.append(uc_tokens)
                        break
            self.unconds = unconds_list
            self.uc_tokens = uc_tokens_list


def unload(self, p):
    if hasattr(self, "handle"):
        hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
        del self.handle


def hook_forward(self, module):
    def forward(
        x,
        context=None,
        mask=None,
        value=None,
        additional_tokens=None,
        *args,
        **kwargs,
    ):
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
            if x.shape[0] == self.batch * 2:
                if self.rev:
                    contn, contp = context.chunk(2)
                    ixn, ixp = x.chunk(2)
                else:
                    contp, contn = context.chunk(2)
                    ixp, ixn = x.chunk(2)  # x[0:self.batch,:,:],x[self.batch:,:,:]

                if conds is not None:
                    if contp.shape[0] != conds.shape[0]:
                        conds = conds.expand(contp.shape[0], -1, -1)
                    contp = torch.cat((contp, conds), 1)
                if unconds is not None:
                    if contn.shape[0] != unconds.shape[0]:
                        unconds = unconds.expand(contn.shape[0], -1, -1)
                    contn = torch.cat((contn, unconds), 1)

                xp = main_forward(
                    self,
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
                xn = main_forward(
                    self,
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

                out = torch.cat([xn, xp]) if self.rev else torch.cat([xp, xn])
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

                return main_forward(
                    self,
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
                concon = counter(self.isxl)
                if context.shape[1] == self.c_len * 77 and concon:
                    if conds is not None:
                        if context.shape[0] != conds.shape[0]:
                            conds = conds.expand(context.shape[0], -1, -1)
                        context = torch.cat([context, conds], 1)
                        tokens = c_tokens
                elif context.shape[1] == self.uc_len * 77 and concon:
                    if unconds is not None:
                        if context.shape[0] != unconds.shape[0]:
                            unconds = unconds.expand(context.shape[0], -1, -1)
                        context = torch.cat([context, unconds], 1)
                        tokens = uc_tokens
                return main_forward(
                    self,
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
            self.conds is not None
            and self.unconds is not None
            and len(self.conds) > 0
            and len(self.unconds) > 0
        ):
            return sub_forward(
                x,
                context,
                mask,
                additional_tokens,
                self.conds[0],
                self.c_tokens[0],
                self.unconds[0],
                self.uc_tokens[0],
            )
        else:
            return sub_forward(
                x, context, mask, additional_tokens, None, None, None, None
            )

    return forward


count = 0
p = True


def counter(isxl):
    global count, p
    count += 1

    limit = 70 if isxl else 16
    outpn = p

    if count == limit:
        p = not p
        count = 0
    return outpn


def main_forward(
    self,
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
    context = default(context, x)
    k = attn.to_k(context)
    if value is not None:
        v = attn.to_v(value)
        del value
    else:
        v = attn.to_v(context)

    if self.active:
        if tokens:
            v[:, -tokens:, :] = -v[:, -tokens:, :]

    out = optimized_attention(q, k, v, attn.heads, mask)
    return attn.to_out(out)


def hook_forwards(self, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            if not remove:
                module.forward = hook_forward(self, module)
            else:
                del module.forward


def resetpcache(p):
    p.cached_c = [None, None]
    p.cached_uc = [None, None]
    p.cached_hr_c = [None, None]
    p.cached_hr_uc = [None, None]


class SdConditioning(list):
    def __init__(
        self,
        prompts,
        is_negative_prompt=False,
        width=None,
        height=None,
        copy_from=None,
    ):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(
            copy_from, "is_negative_prompt", False
        )
        self.width = width or getattr(copy_from, "width", None)
        self.height = height or getattr(copy_from, "height", None)


def hr_dealer(p):
    if not hasattr(p, "hr_prompts"):
        p.hr_prompts = None
    if not hasattr(p, "hr_negative_prompts"):
        p.hr_negative_prompts = None

    return bool(p.hr_prompts), bool(p.hr_negative_prompts)
