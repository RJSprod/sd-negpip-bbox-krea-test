# Regional NegPiP

A fork of [sd-forge-negpip-ClaudeKrea2](https://github.com/RJSprod/sd-forge-negpip-ClaudeKrea2) for Forge [Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which gives a weighted prompt term an **address**.

**NegPiP** lets you write `(man:-1)` in the positive prompt and have the concept subtracted from the image. That is a statement about the whole picture, and there is no way to say *not here*. This adds the missing half:

```
a wide empty landscape at dawn
REGION 0 0 1 0.4 (man:-1)
```

No man in the top four tenths of the frame, and the rest of the picture free to contain one.

> [!IMPORTANT]
> Regions need **Krea 2** on Forge Neo. Every other model family weights the output of a text encoder and has no sequence with an image grid in it to point a box at — on those, this Extension is ordinary NegPiP and the `REGION` lines are ignored with a line on the console saying so.

## The syntax

One region per line, at the start of the line:

```
REGION x0 y0 x1 y1  <terms>
```

- The four numbers are the corners of the box as **fractions of the image** — left, top, right, bottom, in the order a CSS rectangle is written. Fractions rather than pixels because a prompt outlives the resolution it was written at, and highres fix runs its second pass at a different size than its first.
- Separators may be spaces or commas, and a colon after the numbers is optional: `REGION 0.5,0.5, 1, 1: (bird:2.0)` is the same line.
- Corners may be given from either end, and coordinates outside the frame mean the edge.
- `<terms>` is an **ordinary prompt fragment**, weights and all. It is appended to the prompt and encoded with it, so everything that already works keeps working:

| | |
| --- | --- |
| `REGION 0 0 1 0.4 (man:-1)` | that area cannot be a man; everywhere else can |
| `REGION 0 0 1 0.4 (man:-3)` | the same, harder — the weight is a dial, not a switch |
| `REGION 0.6 0 1 1 (a lighthouse:1.4)` | a lighthouse on the right, and nowhere else |
| `REGION 0 0.6 1 1 wildflowers` | no weight at all: a phrase that only applies to the lower third |

A prompt with no `REGION` line behaves exactly as it did before, and the word `REGION` in running text is left alone — what separates a region from prose is the four numbers, not the line break.

> [!TIP]
> **Put `REGION` lines at the end of the prompt.** A prompt does not always reach the text encoder in the shape it was typed — styles, schedules and other Extensions all get a turn, and at least one of them can flatten the line breaks. A region is still found when that happens, but with no line break there is nothing to say where its text ends, so it runs to the end of the line and swallows anything written after it. Last in the prompt, there is nothing to swallow.

## What it actually does

NegPiP carries a term's sign in the **value** projection: the concept is encoded at its magnitude so that attention still routes to it, and every single-stream block multiplies the value of those tokens by a negative number, so the image stream subtracts the concept instead of adding it. One number per text token, the same for every query.

"Only here" is a statement about the **query** — the same key has to be visible to one image patch and invisible to its neighbour — so no per-key quantity can express it, however it is weighted. That query axis is the whole difference between NegPiP and this fork.

Krea 2 makes the geometry free. Its single-stream blocks attend over `[context | refs | img]`, laid out in that order with known lengths, and the image half is a row-major `h × w` grid of patches — the same grid the model numbers its own position ids from. A box in fractions of the image is therefore a set of rows of the attention matrix, by arithmetic, with nothing to infer.

So the region's tokens are appended to the prompt, and the tokens outside the box are stopped from attending to them. Inside the box, the sign applies as it always did. Outside, the concept does not exist — neither added nor subtracted.

### The stage before that one

The single-stream blocks are not the first attention a prompt goes through. `TextFusionTransformer` ends with two blocks of **full self-attention across the token axis**, and a region appended to the prompt is an ordinary token to them: its content is blended into the scene's tokens before the image is attended over at all, and the scene's tokens are read by every patch in the picture.

Masking the boxes downstream cannot undo that. The concept arrives everywhere through the fused scene, at the magnitude it was encoded at and with the sign that is only flipped at the region's own positions — so a negative region reads as a picture violently perturbed and not at all negated.

The same rule is therefore applied one stage earlier: a region's tokens are readable by that region and by nothing else. They still read the scene themselves, so the fragment is encoded knowing what picture it is in; the scene simply does not read them back.

### Two implementations

The `Regional NegPiP` section of **Settings** chooses between them. They compute the same attention, and the test suite asserts they agree to floating-point noise; the choice is only about how much memory it takes to get there.

**`dense`** builds the mask the obvious way: an additive `[B, 1, L, L]` bias, blocked wherever a query outside the box meets one of the region's keys. It is about fifteen lines, it is the definition the other one is checked against, and it works on every attention backend that accepts a mask. It also costs `L²` numbers — around 330 MB at 1536×1536 — most of it structurally zero.

**`merge`** never builds it. The region's keys are a contiguous tail of the text block, so the sequence splits in one place: attention over the global part and over each region separately, combined exactly the way flash attention combines its own chunks, by keeping each part's log-sum-exp. Excluding a query from a region is then setting one entry of a vector rather than a row of a matrix. Two further consequences fall out:

- A prompt in the batch with **no** regions of its own is not split at all and costs exactly what it always cost. That is the negative prompt of nearly every regional generation.
- Each region's part is computed for the queries **inside** its box and nowhere else, so the extra work is proportional to the area the boxes cover rather than to the picture — and flat in how many boxes there are.

**`auto`** (the default) uses `merge` wherever the attention kernel will hand back a log-sum-exp, and `dense` otherwise.

### What it costs

Measured on CPU at a 1536×1536-equivalent sequence (9,350 tokens), against unmasked attention on the same tensors:

| | |
| --- | --- |
| one box over a tenth of the frame | ×1.10 |
| one box over half the frame | ×1.05 |
| one box over the whole frame | ×1.13 |
| four boxes, a tenth of the frame each | ×1.07 |
| `dense`, one box | ×1.16, plus 330 MB of mask |

Those differences are inside the noise of each other, which is the point: at a real sequence length the regional work disappears into the global attention, and adding boxes does not move it. Attention is also only part of a step — the MLP of each block is the larger half — so the effect on a whole generation is smaller again.

## Installing it next to the original

The package, the script, the console prefix, the conditioning keys and every attribute this Extension leaves on a model are named differently from stock **NegPiP**, so both can sit in `extensions/` at once and be A/B tested without moving folders around.

**Only one of them may be enabled.** They wrap the same four methods, and with the markers spelled differently neither recognises the other's wrapper — so the second to arrive wraps the first and every sign is applied twice. This one checks for the other's fingerprint on the model and stands down with a line on the console rather than producing an image that is wrong in a way nobody would connect to having two folders installed.

## When a region does not do anything

Turn on **Regional NegPiP: write a diagnostic log** in Settings. It writes `negpip_regional.log` next to this README, in the Extension's own folder, and every line also goes to the console.

It answers, in order, the questions that are otherwise invisible between the prompt and the pixels:

```
prompt: 1 region(s) parsed
  region 1  box (0.000, 0.000, 0.500, 1.000)  text '(person:-4)'
tokens: region 1 -> 4 token(s) at 130..134 of 134, weight -4.00..-4.00
conditioning: 2 prompt(s), 134 text token(s)
  prompt 1: [130..134] box (0.000, 0.000, 0.500, 1.000)
  prompt 2: no regions
geometry: grid 64x64 = 4096 patches, 0 reference token(s), 4230 in the stream
  region 1 covers 2048 of 4096 patches, rows 0..64, columns 0..32
    ########################........................
    ########################........................
stage: text fusion masked -- 134 token(s), so the scene cannot read a region
stage: image attention -- log-sum-exp merge
attention on region 1:
  inside the box : 0.412% of each patch's attention (kept)
  outside the box: 0.395% of each patch's attention (removed by the mask)
```

- **`prompt as received`** — the prompt exactly as it reached the text encoder, as a `repr`, so a lost line break is visible as one. Compare it against the `batch:` lines above, which are what the Extension was handed before it touched anything: regions present there and gone here means the host's conditioning path flattened them.
- **`no REGION lines`** — the syntax did not match. `REGION` has to be a whole word followed by four numbers.
- **`no sign to apply inside it`** — the box parsed but the term carries no negative weight, usually the `None` emphasis mode.
- **the drawing** — the box as it actually landed on the patch grid. A shape in the wrong place here is a coordinate problem; a shape in the right place means the geometry is fine and the answer is further down.
- **the attention share** — the number that decides whether a region *can* work. NegPiP changes the sign of what a term contributes, and the size of that change is the share of attention the term already had. A term holding a fraction of a per cent of each patch's attention, flipped, moves that patch by a fraction of a per cent, and no weight on the embedding changes that: it is a fact about the attention, not about the emphasis.

The two shares also tell you the mask is real. The outside number is what the region *would* have been worth to a patch outside the box, and the mask takes all of it away; the inside number is what the sign has to work with.

## Console

The Extension reports what it did, so that nothing it decided is silent:

| | |
| --- | --- |
| `NegPiP Regional Loaded` | it is installed and being called, listing the models it can handle |
| `NegPiP Regional Active` | it engaged, naming the model it recognised |
| `NegPiP Regional Enable` | how many tokens are signed, and how many are confined to a box |
| `NegPiP Regional Applied` | the regions reached the transformer, with the patch grid they were built for; `fusion` names the text stage, `merge` or `dense` the image stage |
| `NegPiP Regional Disabled` | it stood down, and why |

Nothing at all for a prompt that contains a `REGION` line or a `(foo:-1.0)` means neither was recognised.

> [!WARNING]
> The weight is applied to the **input embedding** at its magnitude, and the sign is carried separately. `(man:-3)` therefore encodes "man" with three times the usual emphasis and then subtracts it; `(man:-30)` encodes something thirty times over, which is far outside the range the text encoder was trained on and produces a distorted concept rather than a strong one. Stay in single digits.

> [!TIP]
> On Krea 2 the weight is a dial, not a switch. A single-stream transformer attends over the reference and image streams as well, so one text token carries far less of the attention than it does in SD, and `-1.0` can be too subtle to see. Turn it up until the concept goes.

> [!TIP]
> Krea 2 does not work with the `None` **Emphasis Mode**, which reads a weight as literal characters rather than parsing it. The console says so when that happens; every other mode is fine.

## Inherited from NegPiP

Everything the fork was forked from still works:

- `(foo:-1.0)` in the **positive** prompt removes a concept; `(bar:-1.0)` in the **negative** prompt enforces one.
- **SD1**, **SDXL**, **Anima** and **Krea 2** are supported; regions are Krea 2 only.
- Krea 2 normally drops every prompt weight as soon as a reference image is attached. While this Extension is active the weights of an **Edit** prompt are honoured again, so ordinary emphasis takes effect as well.

<table>
    <tr>
        <th>Base</th>
        <th><code>(aqua hair:-1.0)</code><br>in <b>Positive</b> Prompt</th>
        <th><code>(aqua hair:1.5)</code><br>in <b>Negative</b> Prompt</th>
    </tr>
    <tr>
        <td><img src="./img/off.webp" width=256></td>
        <td><img src="./img/negpip.webp" width=256></td>
        <td><img src="./img/neg.webp" width=256></td>
    </tr>
</table>

## Tests

```
python -m pytest tests/
```

They need `torch` and nothing else — no model, no WebUI. What they mostly assert is that the fast path still says what the dense mask says, because that is the half that would break quietly.

## Licence

AGPL-3.0, as the Extension this forked from, and hako-mikan's NegPiP before it.
