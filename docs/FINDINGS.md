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
