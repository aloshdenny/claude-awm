# SynthID-Text detector robustness — measured data

Detector: untrained mean-g scorer. Threshold z = 2.33 (FPR 1%).
Watermark: HF `transformers` SynthID processor, ngram_len=5, depth=30, canonical device-independent sampling table.

Every table includes the `roundtrip` control (unattacked watermarked text). If that row is not well above threshold, the whole table is invalid — this is how two scoring bugs were caught during the study.

## 1 — surface attack ladder vs model scale

### Qwen3.5-0.8B  (84 cells, lengths 1024–8192)

cells below threshold: **0 / 84**  |  min z anywhere: **6.81** (del_40 @ 1024)

| attack | edit% | z@1024 | z@2048 | z@4096 | z@8192 |
|---|---|---|---|---|---|
| roundtrip | 0.0 | 28.3 | 42.0 | 63.8 | 89.1 |
| md_part | 6.8 | 29.9 | 39.2 | 48.8 | 71.0 |
| md_full | 13.6 | 31.7 | 40.4 | 47.1 | 78.1 |
| ws_strip | 1.6 | 28.6 | 41.9 | 57.4 | 89.0 |
| ws_add | 1.6 | 17.9 | 23.8 | 30.3 | 46.9 |
| ws_tabs | 3.1 | 26.8 | 47.9 | 66.5 | 86.9 |
| mix | 15.2 | 24.9 | 30.8 | 40.5 | 83.7 |
| mix_heavy | 15.5 | 10.8 | 15.7 | 19.1 | 33.6 |
| dash | 0.1 | 27.2 | 42.0 | 62.7 | 88.7 |
| amp | 1.5 | 26.0 | 36.0 | 56.5 | 90.4 |
| quotes | 0.0 | 28.3 | 42.0 | 63.8 | 89.1 |
| space | 1.6 | 28.1 | 42.0 | 58.7 | 88.3 |
| contract | 0.1 | 28.2 | 41.8 | 60.6 | 83.8 |
| synonym | 0.2 | 27.5 | 40.1 | 64.2 | 83.9 |
| cosmetic | 3.2 | 24.6 | 35.9 | 51.7 | 93.3 |
| del_01 | 1.1 | 26.9 | 42.9 | 62.7 | 86.7 |
| del_02 | 2.2 | 26.2 | 39.0 | 59.9 | 84.3 |
| del_05 | 4.5 | 25.5 | 34.1 | 49.8 | 73.6 |
| del_10 | 9.1 | 21.5 | 26.3 | 36.9 | 54.4 |
| del_20 | 19.3 | 12.6 | 19.0 | 27.2 | 41.0 |
| del_40 | 40.2 | 6.8 | 11.1 | 12.9 | 17.8 |

### Qwen3.5-4B  (84 cells, lengths 1024–8192)

cells below threshold: **1 / 84**  |  min z anywhere: **2.33** (del_40 @ 1024)

| attack | edit% | z@1024 | z@2048 | z@4096 | z@8192 |
|---|---|---|---|---|---|
| roundtrip | 0.0 | 13.4 | 23.6 | 38.2 | 51.1 |
| md_part | 8.1 | 10.7 | 20.1 | 29.1 | 38.7 |
| md_full | 13.3 | 11.7 | 19.0 | 30.2 | 44.9 |
| ws_strip | 1.3 | 12.6 | 22.6 | 37.5 | 50.5 |
| ws_add | 1.2 | 9.4 | 12.5 | 17.9 | 22.5 |
| ws_tabs | 3.3 | 12.1 | 22.9 | 31.7 | 48.7 |
| mix | 15.6 | 11.0 | 19.3 | 28.5 | 39.2 |
| mix_heavy | 16.7 | 6.6 | 8.9 | 14.5 | 17.2 |
| dash | 0.0 | 13.4 | 23.6 | 37.9 | 50.9 |
| amp | 2.2 | 13.6 | 23.4 | 36.8 | 47.1 |
| quotes | 0.0 | 13.4 | 23.6 | 38.2 | 51.1 |
| space | 1.3 | 12.4 | 23.4 | 34.3 | 50.1 |
| contract | 0.3 | 13.2 | 23.7 | 36.9 | 49.9 |
| synonym | 0.2 | 13.7 | 23.6 | 36.8 | 51.9 |
| cosmetic | 3.5 | 12.9 | 23.5 | 33.1 | 42.9 |
| del_01 | 1.1 | 13.1 | 22.6 | 36.4 | 45.3 |
| del_02 | 2.1 | 13.4 | 23.0 | 35.4 | 43.7 |
| del_05 | 4.4 | 13.8 | 18.8 | 28.8 | 35.9 |
| del_10 | 8.2 | 14.4 | 16.8 | 27.2 | 31.0 |
| del_20 | 18.6 | 9.3 | 9.5 | 13.6 | 25.7 |
| del_40 | 39.4 | 2.3 | 3.1 | 5.3 | 7.6 |

### gpt-oss-20b (native MXFP4)  (126 cells, lengths 1024–32768)

cells below threshold: **0 / 126**  |  min z anywhere: **4.93** (del_40 @ 1024)

| attack | edit% | z@1024 | z@2048 | z@4096 | z@8192 | z@16384 | z@32768 |
|---|---|---|---|---|---|---|---|
| roundtrip | 0.0 | 25.7 | 32.7 | 42.6 | 63.3 | 79.5 | 104.3 |
| md_part | 5.4 | 26.4 | 29.2 | 39.0 | 53.3 | 77.7 | 113.0 |
| md_full | 7.9 | 27.2 | 31.9 | 36.5 | 51.6 | 76.5 | 103.3 |
| ws_strip | 0.9 | 25.6 | 31.5 | 39.8 | 61.2 | 81.3 | 108.2 |
| ws_add | 0.8 | 12.9 | 17.0 | 23.2 | 29.5 | 43.1 | 66.8 |
| ws_tabs | 3.5 | 23.6 | 33.2 | 42.5 | 62.6 | 75.4 | 100.0 |
| mix | 10.2 | 24.1 | 23.6 | 28.8 | 43.4 | 64.2 | 92.8 |
| mix_heavy | 11.9 | 9.0 | 12.8 | 16.1 | 30.1 | 39.9 | 61.6 |
| dash | 0.2 | 26.4 | 31.4 | 39.1 | 61.6 | 80.3 | 113.7 |
| amp | 1.3 | 25.3 | 28.1 | 34.1 | 56.2 | 72.3 | 93.8 |
| quotes | 0.3 | 24.7 | 29.6 | 39.9 | 58.4 | 73.6 | 98.1 |
| space | 0.5 | 27.0 | 34.7 | 42.6 | 61.0 | 78.3 | 105.0 |
| contract | 0.4 | 25.8 | 31.7 | 42.6 | 63.6 | 80.1 | 102.9 |
| synonym | 0.7 | 25.0 | 31.8 | 41.6 | 59.7 | 76.6 | 105.1 |
| cosmetic | 2.4 | 25.9 | 27.8 | 31.1 | 52.8 | 71.3 | 95.8 |
| del_01 | 1.0 | 23.1 | 31.0 | 39.3 | 57.5 | 79.2 | 104.1 |
| del_02 | 2.3 | 23.0 | 30.9 | 38.8 | 60.0 | 83.0 | 112.8 |
| del_05 | 4.4 | 19.5 | 27.9 | 35.6 | 51.9 | 81.6 | 113.5 |
| del_10 | 8.6 | 16.1 | 23.2 | 29.6 | 38.3 | 65.0 | 95.5 |
| del_20 | 18.9 | 11.4 | 14.1 | 19.3 | 26.0 | 42.7 | 75.5 |
| del_40 | 38.2 | 4.9 | 7.4 | 10.9 | 14.0 | 19.4 | 25.4 |

## 2 — watermark strength tracks entropy (code vs prose), Qwen3.5-4B

512-token samples, no attack.

| domain | entropy (bits/tok) | mean-g | z |
|---|---|---|---|
| prose | 1.19 | 0.5408 | 11.1 |
| code | 0.55 | 0.5182 | 5.0 |

ratio code/prose — z 0.45× , entropy 0.46× (they track)
code samples below threshold **with no attack: 3/8**
per-sample code z: [1.7, 4.7, 2.3, 7.7, 3.8, 2.2, 9.1, 8.8]

## 3 — invisible-char stego attacks, RAW vs NORMALIZED, Qwen3.5-4B

### @ 1024 tokens

| attack | edit% | z_raw | z_norm | classification |
|---|---|---|---|---|

### @ 8192 tokens

| attack | edit% | z_raw | z_norm | classification |
|---|---|---|---|---|

## 4 — fidelity (what each attack inserts)

| attack | inserts | Unicode category | invisibility |
|---|---|---|---|
| zwsp_*, zw_word, bidi | zero-width / bidi | Cf (Format) | rigorous — renders nothing |
| nbsp, lineshift, wordshift | spaces | Zs (Space) | rigorous — collapses visually |
| homoglyph | Cyrillic letters | Ll (Letter) | **weaker — font-dependent** |

## 5 — defensive hierarchy (measured)

1. NFKC only → closes nbsp/compat. Leaves zero-width, bidi, homoglyph open.
2. + whitespace-collapse → also closes whitespace-shift. Leaves zero-width, bidi, homoglyph.
3. + strip Cf-category chars → closes zero-width AND bidi. Leaves only homoglyph.
4. + Unicode confusables map (UTS-39) → closes homoglyph. Full coverage.

The intuitive choice (level 1) leaves 3 of 4 families viable. A shipped detector must reach level 4.

## methodology corrections (caught during the study)

- **Device-dependent sampling table:** HF seeds `torch.randint` on the compute device; MPS/CUDA/CPU tables agree only ~50%, so docs made on one device are unscorable on another. Caught by the roundtrip control reading z≈0. Fixed with a canonical CPU-seeded table. All numbers here use it.
- **Null-estimation shadowing:** a nested loop variable collapsed 16 null keysets to 1 (sd≈0, z→millions). Caught by implausible magnitude. Fixed.

## 6 — variation selectors (Mn category), gpt-oss-20b, prose + code @ 8192 tok

Same raw/normalized pairing as section 3, extended with Unicode variation
selectors (U+FE00-FE0F basic plane, U+E0100-E01EF supplementary plane) --
category **Mn** (nonspacing mark), a different Unicode class from the Cf
(format) chars our normalizer strips. Tests whether that's a real gap.

| attack | domain | z_raw | z_norm | edit% |
|---|---|---|---|---|
| roundtrip | prose | 45.03 | 46.51 | 0.0 |
| roundtrip | code | 37.24 | 30.58 | 0.0 |
| zwsp_30 (control, known-defended) | prose | 0.27 | 46.51 | 57.1 |
| homoglyph (known survivor) | prose | 25.75 | 23.77 | 9.0 |
| **vs16_30** | prose | **0.72** | **0.09** | 57.2 |
| **vs16_30** | code | **0.68** | **0.45** | 57.9 |
| vs_supp | code | 2.67 | **1.67** | 22.0 |

**vs16_30 is the first attack all study to cross threshold (z<2.33) both raw
AND after normalization, on both domains.** Unlike zwsp/nbsp/bidi -- which
our normalizer fully reverts to baseline -- variation selectors are meaningful
Unicode codepoints (CJK ideograph variants, emoji presentation selectors),
so NFKC will not and should not fold them away. Closing this gap needs an
explicit strip of the VS ranges, the same class of fix as the Cf-strip we
already do, just a different code-point block.

Code domain again confirms the entropy finding independently: `code` baseline
z is lower than `prose` (37.2 vs 45.0) with no attack, consistent with the
weaker per-token watermark signal on structured/low-entropy generation.

## 7 — Qwen3.8-27B (bf16, H100), prose + code @ 8192 tok

Third model for the variation-selector test. Run on a single H100 at bf16 (no
quantization) after three community quantized checkpoints failed to load
(FP8 needed a torch dtype we don't have; two AWQ/compressed-tensors repacks
had packing-format mismatches).

### prose (baseline z = 35.50)

| attack | z_raw | z_norm | edit% |
|---|---|---|---|
| roundtrip | 35.50 | 35.68 | 0.0 |
| zwsp_30 | -0.09 | **35.68** | 61.3 |
| combo | 0.92 | **35.68** | 46.6 |
| homoglyph | 17.42 | 17.61 | 8.7 |
| **vs16_30** | **-0.67** | **-0.78** | 56.9 |
| vs16 | 3.46 | 3.45 | 23.3 |
| vs_supp | 3.55 | 3.40 | 24.0 |

Replicates the section-6 result on a third model: Cf-category attacks
(zwsp, combo) fully revert to baseline under normalization, while Mn-category
variation selectors stay below threshold both raw and normalized.

### code (baseline z = 4.31) -- NOT a valid attack surface

**The unattacked control is already z=4.31, barely above the 2.33 threshold.**
Attack rows in this domain are therefore uninformative -- you cannot show an
attack defeating a detector that is already near-blind on the clean text. The
meaningful result here is the *baseline itself*:

| model | prose baseline z | code baseline z | ratio |
|---|---|---|---|
| gpt-oss-20b | 45.03 | 37.24 | 0.83 |
| Qwen3.8-27B | 35.50 | **4.31** | **0.12** |

This is the strongest entropy-effect observation in the study. Qwen3.8-27B's
code generation is so low-entropy that the watermark is nearly undetectable
with no adversary involved at all. Whether this is an entropy effect alone or
partly an artifact of this model's code-generation style (very templated
output, high token predictability) is not something these two documents can
separate -- it needs a per-token entropy measurement like section 2, which
was not run for this model.

## 8 — DeepSeek-R1-Distill-Qwen-14B (4-bit MLX, Apple Silicon)

Fifth model, and the first reasoning-distilled one. Run locally on an M-series
Mac (17GB unified memory) through the MLX bridge in `src/synthid_mlx.py`,
because llama.cpp/Ollama GGUF builds cannot be watermarked at all -- they do
not expose logits during generation, which is where the watermark is applied.
The MLX fast path was verified bit-identical to the HF reference before
generating (g-values exact; reweight max prob diff 2.98e-08).

| metric | value |
|---|---|
| quantization | 4-bit, 8.3 GB |
| throughput | ~6.3 tok/s |
| tokens / scoreable | 4096 / 3955 |
| mean g | 0.5259 |
| null mean / sd | 0.5000 / 0.00146 |
| **z** | **17.66** (threshold 2.33) |

Detected comfortably. But note the mean g: **0.526, against 0.56-0.60 for
ordinary prose on the other models.** The generated stream turned out to be
almost entirely `<think>` reasoning chains ("Okay, I need to... Hmm... Wait...
Oh right..."), with six closing `</think>` tags and no substantial final-answer
text. So this row measures the watermark on **reasoning traces**, not on
answers.

### the reasoning-entropy hypothesis, and why it was wrong

The obvious explanation was that reasoning text is more templated and thus
lower-entropy than prose, leaving the tournament less room to bias each
choice. **That was measured and it is false.** Per-token entropy on the same
model, same run, splitting on the `</think>` boundary:

| span | tokens | entropy |
|---|---|---|
| reasoning (`<think>`) | 1171 | **0.891** bits/tok |
| final answer | 1246 | **0.441** bits/tok |

Reasoning is the **higher**-entropy half, not the lower one. A first attempt
at this measurement was confounded and is reported here for completeness: it
used short factual prompts ("what is 17x23?") chosen so reasoning would fit
the token budget, which made the answers deterministic strings and gave a
meaningless 0.41x ratio. Re-run with prompts demanding substantive prose
answers, the direction held at 0.49x.

So the depressed `mean_g` is not a reasoning-vs-answer effect. The better
explanation is model-level: R1-distill runs low-entropy across **both** spans
(weighted overall 0.659 bits/tok) against 1.19 bits/tok measured for
Qwen3.5-4B prose in section 2, roughly 0.55x. A distilled, 4-bit-quantized
model is simply more confident per token than the models in the earlier runs.
That is still a cross-model comparison and not a controlled one.

Caveat on the numbers above: the two prompts contributed very unevenly. The
first supplied nearly all the reasoning tokens, the second nearly all the
answer tokens, so this is closer to a cross-prompt comparison than a clean
within-prompt one. Directionally consistent across both attempts, but it
would want more prompts to be called settled.

**Detector note.** Nothing about the detector changed for this model. It has
no learned parameters: `ngram_len`, `depth`, a keyed LCG hash chain, a
seed-0 sampling table, and a 16-wrong-key empirical null. Each of the five
models tested is an independent generator checked against that same fixed
algorithm. The only model-specific requirement is the tokenizer, since
detection must reproduce the exact token IDs generation produced.

## 9 — gpt-oss-120b: why it needs an 80 GB card

Briefly, because it shapes section 10: the 4-bit MLX build is 65.8 GB against
17.2 GB of unified memory on the Mac, a 3.8x oversubscription. `mlx_lm.load`
defaults to `lazy=False` and materialises everything at once, which the OS
kills; with `lazy=True` the load returns instantly at 0 GB but generation is
killed before the first token, because one forward pass across 36 layers plus
KV cache does not fit regardless of MoE sparsity. 4-bit is the smallest
published quantisation, so there is no smaller fallback. The model needs a
machine that can hold it, which is what section 10 uses.


## 10 — gpt-oss-120b on an H100

Same model, same watermark, different machine. On Modal with a single H100
(80 GB) it loads **native MXFP4 at 65.3 GB resident** with room to spare, and
all three domains watermark and detect cleanly:

| domain | tokens | scored | mean g | z | tok/s |
|---|---|---|---|---|---|
| prose | 2048 | 1989 | 0.5293 | **16.09** | 10.3 |
| code | 2048 | 1954 | 0.5194 | **11.97** | 14.0 |
| reasoning | 2268 | 2047 | 0.5207 | **11.61** | 13.2 |

Threshold 2.33; all three detected. Roughly 15-30x the throughput the Mac
would have managed had it run at all, and it completed in minutes rather than
the 14+ hours estimated locally.

**The load figure is the thing to check, not the model size.** 65.3 GB
resident confirms native MXFP4. With Triton absent, transformers silently
dequantizes those weights to bf16 (~240 GB for this model) and OOMs -- the
same trap noted earlier in the multi-GPU work. Installing `triton` and
`kernels` in the image is what makes an 80 GB card sufficient.

The domain ordering matches the entropy finding: **prose (16.09) carries a
clearly stronger mark than code (11.97) or reasoning (11.61)**. Worth noting
all three `mean_g` values cluster near 0.52, below the 0.56-0.60 seen on
smaller models' prose, so the per-token signal here is thinner across the
board and the high z comes partly from token count.


## 11 — gpt-oss-120b across context length (2k / 8k / 32k)

One 32k generation per domain; the 2k and 8k rows are scored *prefixes* of
that same stream, which is how the original ladder worked and costs nothing
extra.

| domain | z @2k | z @8k | z @32k |
|---|---|---|---|
| prose | 16.09 | 28.01 | **30.67** |
| code | 11.97 | 14.83 | **19.20** |
| reasoning | 9.58 | 15.44 | **16.86** |

All nine cells detected (threshold 2.33). Two things worth reading off this:

**z grows with length, `mean_g` does not.** Within each domain `mean_g` sits
flat near 0.52 at every length (prose 0.5293 / 0.5259 / 0.5258). The whole z
increase comes from token count, exactly as the sqrt(n) scaling predicts. Per-token
signal strength is a property of the model and domain; length only buys
statistical confidence.

**The domain ordering holds at every length**: prose > code ~ reasoning,
consistent with the entropy finding. Length does not rescue a low-entropy
domain relative to a high-entropy one, it lifts both.

This is the practical restatement of the earlier "length helps the detector,
not the attacker" result, now measured on the largest model in the study.

## 12 — reasoning tokens dominate watermark strength on low-entropy domains

This one came out of a mistake. A Modal re-run of Qwen3.8-27B on code was
meant to extend the length ladder to 32k, but it used the tokenizer's default
chat template instead of the study harness's `enable_thinking=False`. Those
are different experiments: the default template ends `<think>\n` and the model
reasons before answering, while the harness ends `<think>\n\n</think>\n\n` and
the model emits code directly.

The accident produced a controlled comparison at matched length (8192 tokens,
same model, same domain, both bf16 on an H100):

| condition | mean g | z @8192 |
|---|---|---|
| thinking **off** (pure code) | 0.5056 | **4.31** |
| thinking **on** (reasoning, then code) | 0.5374 | **25.53** |

**One flag moves z by roughly 6x.** And the cross-domain picture sharpens it:

| condition | prose mean g | code mean g |
|---|---|---|
| thinking off | (z 35.50) | 0.5056 (z 4.31) |
| thinking on | 0.5406 | 0.5395 |

With thinking enabled, prose and code have *nearly identical* per-token signal
(0.5406 vs 0.5395). With it disabled they diverge enormously. The reasoning
preamble dominates the token stream and **homogenises entropy across domains**.

The practical consequence: **"watermark strength on code" is not well defined
for a reasoning model without specifying thinking mode.** In default mode a
"code" request produces mostly reasoning tokens, which are ordinary prose and
carry the mark normally. The low-entropy weakness in section 2 applies to the
*emitted code itself*, and a model that thinks out loud first largely escapes
it -- not by strengthening the watermark on code, but by emitting less code
per response.

With thinking on, the length ladder behaves normally: z = 17.67 / 25.53 /
43.87 at 2k / 8k / 32k.

**Caveat:** the prompt sets differ slightly between the two runs (eight study
prompts vs three here, overlapping but not identical), so the contrast is
strong but not perfectly controlled. The `mean_g` gap (0.5056 vs 0.5374) is
far larger than prompt variation plausibly explains, but a purpose-built A/B
on identical prompts would settle it.


## 13 — the attack is not context-agnostic: required insertion rate rises with length

A 240-cell benchmark on gpt-oss-20b to answer directly: does random
variation-selector insertion *always* defeat the detector? Grid = 2 domains x
4 lengths x 5 insertion rates x 8 independent attack seeds, each scored raw
and normalized against a 16-wrong-key null.

**It does not always work. Success is a function of insertion rate, and the
rate you need grows with context length.** Code domain, cells below threshold
out of 8 random seeds:

| insertion rate | @1024 | @2048 | @4096 | @8192 |
|---|---|---|---|---|
| 2% | 0/8 | 0/8 | 0/8 | 0/8 |
| 5% | 1/8 | 0/8 | 0/8 | 0/8 |
| 10% | 7/8 | 2/8 | **0/8** | **0/8** |
| 20% | 8/8 | 8/8 | 7/8 | 7/8 |
| 30% | 8/8 | 8/8 | 8/8 | 8/8 |

Prose (only long enough for 1k/2k here) shows the same shape: 10% gives 5/8 at
1024 but 1/8 at 2048.

Two things this settles:

**The seed is not the variable that matters; the rate is.** At 20-30% every
one of the 8 independent random placements succeeds. Randomness of insertion
is not the weak link -- a fixed rate behaves consistently across seeds. So
"random insertion works" is true *above a rate threshold* and false below it.

**That threshold climbs with length, in the detector's favour.** The detector
accumulates signal like sqrt(n), so a longer document has more margin to burn
through. 10% clears 1024 tokens and fails entirely by 4096. Even 20% starts
slipping at 4k+. Only 30% held across everything tested (to 8k), and nothing
here rules out that 16k/32k needs more still.

Overall: **110/240 attack cells (46%) fell below threshold.** A blanket claim
that the attack "always works" or is "context/architecture-agnostic" is
refuted by its own success curve. The honest statement is a dose-response
relationship: it works reliably only when insertion rate exceeds a
length-dependent threshold (roughly >=20% up to a few thousand tokens, >=30%
beyond), at which point the invisible-character load on the text is
substantial even though it stays visually identical.

Normalization tracked raw closely throughout (121/240 vs 110/240 below
threshold), consistent with the Mn-category gap holding across the whole grid
rather than at a single point.
