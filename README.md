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

A prompt with no `REGION` line behaves exactly as it did before, and the word `REGION` in running text is left alone — the line has to start with it.

## What it actually does

NegPiP carries a term's sign in the **value** projection: the concept is encoded at its magnitude so that attention still routes to it, and every single-stream block multiplies the value of those tokens by a negative number, so the image stream subtracts the concept instead of adding it. One number per text token, the same for every query.

"Only here" is a statement about the **query** — the same key has to be visible to one image patch and invisible to its neighbour — so no per-key quantity can express it, however it is weighted. That query axis is the whole difference between NegPiP and this fork.

Krea 2 makes the geometry free. Its single-stream blocks attend over `[context | refs | img]`, laid out in that order with known lengths, and the image half is a row-major `h × w` grid of patches — the same grid the model numbers its own position ids from. A box in fractions of the image is therefore a set of rows of the attention matrix, by arithmetic, with nothing to infer.

So the region's tokens are appended to the prompt, and the tokens outside the box are stopped from attending to them. Inside the box, the sign applies as it always did. Outside, the concept does not exist — neither added nor subtracted.

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

## Console

The Extension reports what it did, so that nothing it decided is silent:

| | |
| --- | --- |
| `NegPiP Regional Loaded` | it is installed and being called, listing the models it can handle |
| `NegPiP Regional Active` | it engaged, naming the model it recognised |
| `NegPiP Regional Enable` | how many tokens are signed, and how many are confined to a box |
| `NegPiP Regional Applied` | the regions reached the transformer, with the patch grid they were built for |
| `NegPiP Regional Disabled` | it stood down, and why |

Nothing at all for a prompt that contains a `REGION` line or a `(foo:-1.0)` means neither was recognised.

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
