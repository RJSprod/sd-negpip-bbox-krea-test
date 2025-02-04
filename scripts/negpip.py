import re

from modules import scripts
from modules.prompt_parser import get_learned_conditioning
from modules.prompt_parser import get_learned_conditioning_prompt_schedules as get_ps
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from torch import cat

from lib_negpip.utils import (
    SdConditioning,
    hook_forwards,
    hr_dealer,
    resetpcache,
    unload,
)

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
        self.rev = p.sampler_name not in ("DDIM", "PLMS", "UniPC")

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
                    padtextweight = []
                    pad = 0

                    for sep_prompt in sep_prompts:
                        neg_targets = []
                        text_weights = []
                        neg_matches = re.findall(neg_pattern, sep_prompt)

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

        scheduled_p = get_ps(p.prompts, p.steps)
        nip = getScheduledNegs(scheduled_p, p.prompts)

        scheduled_np = get_ps(p.negative_prompts, p.steps)
        pin = getScheduledNegs(scheduled_np, p.negative_prompts)

        if self.hrp:
            scheduled_hr_p = get_ps(
                p.hr_prompts,
                p.hr_second_pass_steps if p.hr_second_pass_steps > 0 else p.steps,
            )
            hr_nip = getScheduledNegs(scheduled_hr_p, p.hr_prompts)

        if self.hrn:
            scheduled_hr_np = get_ps(
                p.hr_negative_prompts,
                p.hr_second_pass_steps if p.hr_second_pass_steps > 0 else p.steps,
            )
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
                    [f"({target[0]}:{-target[1]})"],
                    width=p.width,
                    height=p.height,
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

                token, token_len = tokenizer(target[0])
                conds.append(
                    cond[0][0].cond[1 : token_len + 2, :]
                    if not self.isxl
                    else cond[0][0].cond["crossattn"][1 : token_len + 2, :]
                )

            conds = cat(conds, 0)
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

        def calcChunks(a, b):
            return a // b if a % b == 0 else a // b + 1

        self.c_len = calcChunks(tokenizer(p.prompts[0])[1], 75)
        self.uc_len = calcChunks(tokenizer(p.negative_prompts[0])[1], 75)

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
