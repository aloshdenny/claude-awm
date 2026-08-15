# claude-awm: can you scrub a SynthID text watermark by editing the text?

Hola, Alosh here ✌🏻

Short answer, from ~500 measured attack/length cells across four models: **no, not with any surface edit that keeps the text readable.** The one mechanical weakness I found dies to a single line of input normalization. The place the watermark is *actually* weak has nothing to do with attacks at all — it's low-entropy text like code, where the mark is thin by construction and no editing is needed.

So this repo is the opposite of what the meme at the bottom implies. It's a robustness study that mostly failed to break the thing, and the failures are the interesting part.

## what this is

Anthropic (and Google DeepMind before them, the [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) paper) watermark generated text by biasing token sampling with a keyed tournament. The signal lives in *which tokens got picked*, not in any hidden character. I wanted to know how fragile that is to a motivated person with a find-and-replace, so I built the generator + an untrained mean-g detector and threw attacks at it.

Detector threshold is z = 2.33 (1% false positive rate). Above that = watermark detected. I generated watermarked text, attacked it, re-tokenized, and scored. Every results table carries a `roundtrip` control (unattacked watermarked text) so you can see the detector is actually working in that row — twice this study a bug made every z collapse to zero and the control is the only thing that caught it.

## the thing I was actually worried about

My starting fear was the boring one: that em-dashes and "and"→"&" and stripping markdown would just work, and the watermark would turn out to be security theater. A lot of people online assume exactly that.

It doesn't work. Here's the whole attack ladder on gpt-oss-20b, out to 32k tokens:

| attack | edit rate | z @ 1k | z @ 32k |
|---|---|---|---|
| roundtrip (control) | 0% | 25.7 | 104.3 |
| em-dash → hyphen | ~0% | 26.4 | 113.7 |
| strip **all** markdown | 13.6% | 27.2 | 103.3 |
| AmE→BrE + abbreviations | 1.3% | ~28 | ~100 |
| delete 40% of every word | 38% | 4.9 | 25.4 |

Across three models (0.8B, 4B, 20B = 294 cells), exactly **one** cell crossed threshold: deleting 39% of every word at the shortest length on the 4B. That's not an attack, that's shredding the document. Everything readable stays detected by a wide margin.

Two things surprised me:

- **Edit *count* doesn't predict damage — edit *geometry* does.** Stripping all markdown (13.6% of tokens) did nothing; it even scored slightly *above* baseline. Injecting stray spaces at 1.6% did 25× more damage per edit. Markdown markers cluster, so their corruption windows overlap and the long prose runs between them keep replaying the watermark seed intact. Scattered edits that desync the tokenizer hit fresh windows every time.
- **Length helps the detector, not the attacker.** z grows like √(tokens). "Fool it over a long context" is backwards — 32k is the hardest case to attack, not the easiest.

Full mechanism and per-attack tables in [docs/FINDINGS.md](docs/FINDINGS.md).

## the one real mechanical weakness (and why it doesn't count)

The strongest attack that keeps text readable is **invisible-character injection** — zero-width spaces, non-breaking spaces, bidi controls. These shift where the BPE tokenizer puts its boundaries, which desyncs the seed chain. Raw, `zwsp` drives the 4B's z from 11.4 to **−0.7**. Undetectable.

Then you run one line of normalization on the input — strip format-category chars, NFKC fold, collapse whitespace — and it snaps right back to 10.7. Every zero-width / nbsp / whitespace-shift attack fully recovers. I checked all of them raw *and* post-normalize; the gap is the whole point.

**One survivor:** homoglyphs (Cyrillic а for Latin a). NFKC doesn't fold script-confusables, so z stays down (8.0 → 8.2 normalized). That's the single stego vector a standard normalizer misses, and the fix is a Unicode confusables map (UTS-39), not NFKC. If there's a defensive takeaway for whoever ships one of these detectors, it's: normalize format chars *and* keep a confusables skeleton, because NFKC alone leaves 2 of the families I tested open.

Every attack here is verified visually identical to the original — see [src/fidelity_check.py](src/fidelity_check.py). Four families insert only Unicode format (Cf) chars that render nothing; homoglyphs are the honest exception (font-dependent, not guaranteed invisible).

## the finding that actually matters

The watermark rides on the model's per-token uncertainty. Where the model is confident about the next token, the tournament has no room to bias it, so no signal goes in. That means **the mark is weak on low-entropy text — and code is low-entropy.**

Qwen3.5-4B, prose vs code, 512-token samples, no attack at all:

| domain | entropy | z |
|---|---|---|
| prose | 1.19 bits/tok | 11.1 |
| code | 0.55 bits/tok | 5.0 |

z ratio 0.45×, entropy ratio 0.46× — they move together, which is the mechanism showing through. And **3 of 8 code samples fell to or below the detection threshold on their own.** The tightest one (a bare algorithm, 0.2 bits/token) scored 1.7 — a miss. No editing, no tricks, just ask for code.

This is the part I'd lead with if I were reporting it, because it needs no adversary and no assumption that the detector is misconfigured. It says: a single confidence threshold across domains is unsafe, and short code snippets are close to unwatermarkable. Generalizes to JSON, config, structured extraction, boilerplate.

## what's wrong with this / what I didn't get to

Being honest about the gaps, grayswan-style.

- **GLM-5.2 produced zero data.** I rented an 8×A100 pod for it and hit five infra failures in a row (deprecated download cmd, torch/torchvision ABI breaks, the model loading into host RAM instead of the GPUs), burned ~$25 including $19 on a pod that idled because I trusted a download that never started, and terminated it with nothing. Kimi-K3 never got attempted. The frontier-scale question is open.
- **The entropy + stego experiments only ran on the 4B.** The 0.8B and 20B have the full attack ladder but not the code-vs-prose or invisible-char runs. So the entropy finding is one model, cleanly, not three.
- **The detector is the untrained mean-g scorer, not the trained Bayesian one** from the paper. The Bayesian detector would likely be *more* sensitive, so these z values are a floor, but I didn't measure it.
- **My homoglyph fidelity claim is "typical reader," not proven.** Cyrillic а is category Ll, so its invisibility is a font property, not a Unicode guarantee.
- **n is small** — 2 documents per cell for the ladders, 8 samples per domain for entropy. Enough for the effect sizes here (they're large), not enough for tight per-attack error bars.
- I don't provide a tuned evasion recipe and that's deliberate. The point was to measure where the frontier is, not to package a bypass.

## layout

```
src/synthid_robustness.py   generator + attack ladder + mean-g detector + normalizer
src/code_vs_prose.py        the entropy experiment (with per-token entropy tap)
src/fidelity_check.py       proves the stego attacks are visually identical
src/synthid_mlx.py          watermarking bridge for Apple Silicon (MLX), validated vs HF
src/prompts_code.py         prose vs code prompt sets
src/build_report_data.py    assembles results/ into the tables in FINDINGS.md
results/                    the JSON this is all computed from (0.8B, 4B, 20B, entropy, stego)
docs/FINDINGS.md            every table, the defense hierarchy, the bugs I caught
```

## running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

```bash
# generate watermarked docs + run the full attack ladder on a model
MODEL=Qwen/Qwen3.5-4B LENGTHS=1024,2048,4096,8192 N_DOCS=2 \
  DOCS=docs_4b.json OUT=res_4b.json python src/synthid_robustness.py

# the entropy experiment (code vs prose)
python src/code_vs_prose.py --model Qwen/Qwen3.5-4B --out res_cvp_4b.json

# the stego attacks, scored raw AND post-normalization
ATTACK_SET=desync DEFENSE=1 SCORE_ONLY=1 MODEL=Qwen/Qwen3.5-4B \
  DOCS=docs_4b.json OUT=res_defense.json python src/synthid_robustness.py
```

Watermarking needs the full next-token distribution, so it runs through `transformers` (CUDA native MXFP4, or MPS/CPU). Ollama/llama.cpp can't do it — they don't expose logits mid-generation. On Apple Silicon, `src/synthid_mlx.py` bridges MLX generation into the watermark math; it's validated bit-identical to the HF reference.

---

![the dream, allegedly](assets/dewatermarked.png)

*(the meme that started it. the study says the bill is still watermarked.)*

---

Note: I used Claude Code heavily for the implementation and to rerun experiments across three machines (a Mac, a rented 3090, my own 4090). The experiment design, the attacks I wanted tried, and the calls on framing are mine. Claude also, for what it's worth, kept refusing to tune the attacks into an actual working bypass and insisted on measuring every attack against its own defense — which is why the stego section has a raw *and* a normalized column instead of just the raw one.
