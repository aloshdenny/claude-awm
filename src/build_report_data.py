"""
Assembles results/*.json into the markdown tables in docs/FINDINGS.md.
Run from repo root:  python src/build_report_data.py > docs/FINDINGS.md
Data only -- no narrative. Every table carries the roundtrip control.
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
THRESH = 2.33


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def cells(d):
    return {(x["length"], x["attack"]): x for x in d["results"]}


print("# SynthID-Text detector robustness — measured data\n")
print("Detector: untrained mean-g scorer. Threshold z = 2.33 (FPR 1%).")
print("Watermark: HF `transformers` SynthID processor, ngram_len=5, depth=30, "
      "canonical device-independent sampling table.\n")
print("Every table includes the `roundtrip` control (unattacked watermarked text). "
      "If that row is not well above threshold, the whole table is invalid — "
      "this is how two scoring bugs were caught during the study.\n")

# ---- attack ladder, all three models ----
print("## 1 — surface attack ladder vs model scale\n")
for fn, name in [("res_08b_full.json", "Qwen3.5-0.8B"),
                 ("res_4b.json", "Qwen3.5-4B"),
                 ("res_20b.json", "gpt-oss-20b (native MXFP4)")]:
    d = load(fn)
    if not d:
        continue
    m = cells(d)
    Ls = sorted({k[0] for k in m})
    A = [a for a in dict.fromkeys(x["attack"] for x in d["results"])]
    below = sum(1 for x in m.values() if x["z"] < THRESH)
    mn = min(m.values(), key=lambda x: x["z"])
    print(f"### {name}  ({len(m)} cells, lengths {Ls[0]}–{Ls[-1]})\n")
    print(f"cells below threshold: **{below} / {len(m)}**  |  "
          f"min z anywhere: **{mn['z']:.2f}** ({mn['attack']} @ {mn['length']})\n")
    head = "| attack | edit% | " + " | ".join(f"z@{L}" for L in Ls) + " |"
    print(head)
    print("|" + "---|" * (len(Ls) + 2))
    for a in A:
        er = m[(Ls[0], a)]["edit_rate"] * 100
        row = " | ".join(f"{m[(L, a)]['z']:.1f}" for L in Ls)
        print(f"| {a} | {er:.1f} | {row} |")
    print()

# ---- entropy ----
cvp = load("res_cvp_4b.json")
if cvp:
    print("## 2 — watermark strength tracks entropy (code vs prose), Qwen3.5-4B\n")
    print("512-token samples, no attack.\n")
    print("| domain | entropy (bits/tok) | mean-g | z |")
    print("|---|---|---|---|")
    for dom in ["prose", "code"]:
        x = cvp[dom]
        print(f"| {dom} | {x['entropy']:.2f} | {x['mean_g']:.4f} | {x['z']:.1f} |")
    zr = cvp["code"]["z"] / cvp["prose"]["z"]
    er = cvp["code"]["entropy"] / cvp["prose"]["entropy"]
    below = sum(1 for z in cvp["code"]["per_z"] if z < THRESH)
    print(f"\nratio code/prose — z {zr:.2f}× , entropy {er:.2f}× (they track)")
    print(f"code samples below threshold **with no attack: {below}/"
          f"{len(cvp['code']['per_z'])}**")
    print(f"per-sample code z: {[round(z,1) for z in cvp['code']['per_z']]}\n")

# ---- stego raw vs normalized ----
df = load("res_defense2_4b.json")
if df:
    print("## 3 — invisible-char stego attacks, RAW vs NORMALIZED, Qwen3.5-4B\n")
    m = cells(df)
    base = {L: m[(L, "roundtrip")].get("z_norm") for L in {k[0] for k in m}}
    order = ["roundtrip", "zwsp_10", "zwsp_30", "zw_word", "nbsp",
             "bidi", "homoglyph", "lineshift", "wordshift", "combo"]
    for L in sorted({k[0] for k in m}):
        print(f"### @ {L} tokens\n")
        print("| attack | edit% | z_raw | z_norm | classification |")
        print("|---|---|---|---|---|")
        bn = base[L]
        for a in order:
            if (a, L) not in m:
                continue
            x = m[(a, L)]
            zr, zn = x["z"], x.get("z_norm", float("nan"))
            if a == "roundtrip":
                cls = "control"
            elif zr > bn * 0.85:
                cls = "attack ineffective raw"
            elif zn > bn * 0.85:
                cls = "normalization recovers"
            else:
                cls = "**SURVIVES normalization**"
            print(f"| {a} | {x['edit_rate']*100:.1f} | {zr:.1f} | {zn:.1f} | {cls} |")
        print()
    print("## 4 — fidelity (what each attack inserts)\n")
    print("| attack | inserts | Unicode category | invisibility |")
    print("|---|---|---|---|")
    print("| zwsp_*, zw_word, bidi | zero-width / bidi | Cf (Format) | rigorous — renders nothing |")
    print("| nbsp, lineshift, wordshift | spaces | Zs (Space) | rigorous — collapses visually |")
    print("| homoglyph | Cyrillic letters | Ll (Letter) | **weaker — font-dependent** |\n")
    print("## 5 — defensive hierarchy (measured)\n")
    print("1. NFKC only → closes nbsp/compat. Leaves zero-width, bidi, homoglyph open.")
    print("2. + whitespace-collapse → also closes whitespace-shift. Leaves zero-width, bidi, homoglyph.")
    print("3. + strip Cf-category chars → closes zero-width AND bidi. Leaves only homoglyph.")
    print("4. + Unicode confusables map (UTS-39) → closes homoglyph. Full coverage.\n")
    print("The intuitive choice (level 1) leaves 3 of 4 families viable. A shipped "
          "detector must reach level 4.\n")

print("## methodology corrections (caught during the study)\n")
print("- **Device-dependent sampling table:** HF seeds `torch.randint` on the compute "
      "device; MPS/CUDA/CPU tables agree only ~50%, so docs made on one device are "
      "unscorable on another. Caught by the roundtrip control reading z≈0. Fixed with a "
      "canonical CPU-seeded table. All numbers here use it.")
print("- **Null-estimation shadowing:** a nested loop variable collapsed 16 null keysets "
      "to 1 (sd≈0, z→millions). Caught by implausible magnitude. Fixed.")
