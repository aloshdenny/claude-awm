"""
Generate the two study charts as self-contained theme-matched SVG, from the
committed results. Run from repo root:  python site/gen_charts.py
Writes site/charts.html (and docs/charts.html for GitHub Pages).

Chart 1: watermark strength (z) rises with context length -- the detector's
         advantage grows ~sqrt(n). gpt-oss-120b, three domains.
Chart 2: the attack has to work harder as context grows -- success rate over
         (context length x insertion rate). gpt-oss-20b, 240-cell benchmark.
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results")

# ---- palette (matches the site: true black, SF font, purple accent) ----
BG, INK, SOFT, FAINT, RULE = "#000000", "#ffffff", "#a0a0a0", "#6a6a6a", "#222222"
UV, UV2 = "#a78bfa", "#c4b5fd"
FLAG, CLEAR = "#f0a850", "#5ec7a0"
FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif"
MONO = "ui-monospace, 'SF Mono', SFMono-Regular, Menlo, monospace"


def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;")


# =========================================================================
# CHART 1 -- z vs context length
# =========================================================================
def chart1():
    d = json.load(open(os.path.join(RES, "res_120b_lengths.json")))["cells"]
    doms = [("prose", UV2), ("code", UV), ("reasoning", FLAG)]
    lengths = [2048, 8192, 32768]
    series = {dom: [(L, d[f"{dom}@{L}"]["z"]) for L in lengths] for dom, _ in doms}

    W, H = 620, 380
    ml, mr, mt, mb = 58, 130, 30, 52
    pw, ph = W - ml - mr, H - mt - mb
    zmax = 34
    # log2 x scale over 2048..32768
    def xf(L): return ml + (math.log2(L) - math.log2(2048)) / (math.log2(32768) - math.log2(2048)) * pw
    def yf(z): return mt + ph - (z / zmax) * ph

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">']
    # y gridlines + labels
    for z in [0, 10, 20, 30]:
        y = yf(z)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{RULE}" stroke-width="1"/>')
        s.append(f'<text x="{ml-10}" y="{y+4:.1f}" fill="{FAINT}" font-size="11" font-family="{MONO}" text-anchor="end">{z}</text>')
    # threshold
    yt = yf(2.33)
    s.append(f'<line x1="{ml}" y1="{yt:.1f}" x2="{ml+pw}" y2="{yt:.1f}" stroke="{CLEAR}" stroke-width="1" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{ml+pw}" y="{yt-6:.1f}" fill="{CLEAR}" font-size="10" font-family="{MONO}" text-anchor="end">detection threshold z=2.33</text>')
    # x labels
    for L in lengths:
        x = xf(L)
        s.append(f'<text x="{x:.1f}" y="{mt+ph+20}" fill="{SOFT}" font-size="11" font-family="{MONO}" text-anchor="middle">{L//1024}k</text>')
    s.append(f'<text x="{ml+pw/2:.1f}" y="{H-8}" fill="{FAINT}" font-size="11" text-anchor="middle">context length (tokens)</text>')
    s.append(f'<text x="16" y="{mt+ph/2:.1f}" fill="{FAINT}" font-size="11" text-anchor="middle" transform="rotate(-90 16 {mt+ph/2:.1f})">detector z-score</text>')
    # series
    for i, (dom, col) in enumerate(doms):
        pts = series[dom]
        path = " ".join(f"{'M' if j==0 else 'L'}{xf(L):.1f} {yf(z):.1f}" for j,(L,z) in enumerate(pts))
        s.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for L, z in pts:
            s.append(f'<circle cx="{xf(L):.1f}" cy="{yf(z):.1f}" r="3.5" fill="{col}"/>')
        lx, lz = pts[-1]
        s.append(f'<text x="{xf(lx)+10:.1f}" y="{yf(lz)+4:.1f}" fill="{col}" font-size="12" font-family="{MONO}">{dom}</text>')
    s.append('</svg>')
    return "".join(s)


# =========================================================================
# CHART 2 -- success rate over (length x insertion rate)
# =========================================================================
def chart2():
    rows = [r for r in json.load(open(os.path.join(RES, "res_vs_bench_20b.json"))) if r["kind"] == "vs16"]
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])  # (L,rate) -> [succ,total]
    for r in rows:
        if r["domain"] != "code":
            continue
        a = agg[(r["length"], r["rate"])]; a[1] += 1
        if r["z_raw"] < 2.33: a[0] += 1
    lengths = [1024, 2048, 4096, 8192]
    rates = [0.02, 0.05, 0.10, 0.20, 0.30]

    W, H = 620, 380
    ml, mr, mt, mb = 62, 96, 26, 66
    cw = (W - ml - mr) / len(rates)
    chh = (H - mt - mb) / len(lengths)

    def ramp(f):  # 0..1 -> dark to purple
        if f <= 0: return "#0e0e12"
        r0, g0, b0 = 14, 14, 18
        r1, g1, b1 = 167, 139, 250
        return f"#{int(r0+(r1-r0)*f):02x}{int(g0+(g1-g0)*f):02x}{int(b0+(b1-b0)*f):02x}"

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">']
    for li, L in enumerate(lengths):
        y = mt + li * chh
        s.append(f'<text x="{ml-10}" y="{y+chh/2+4:.1f}" fill="{SOFT}" font-size="11" font-family="{MONO}" text-anchor="end">{L//1024}k</text>')
        for ri, rate in enumerate(rates):
            x = ml + ri * cw
            succ, tot = agg[(L, rate)]
            f = succ / tot if tot else 0
            s.append(f'<rect x="{x+1:.1f}" y="{y+1:.1f}" width="{cw-2:.1f}" height="{chh-2:.1f}" fill="{ramp(f)}" rx="2"/>')
            tc = "#000000" if f > 0.55 else SOFT
            s.append(f'<text x="{x+cw/2:.1f}" y="{y+chh/2+4:.1f}" fill="{tc}" font-size="11" font-family="{MONO}" text-anchor="middle">{succ}/{tot}</text>')
    # x labels (insertion rate)
    for ri, rate in enumerate(rates):
        x = ml + ri * cw + cw / 2
        s.append(f'<text x="{x:.1f}" y="{mt+len(lengths)*chh+20:.1f}" fill="{SOFT}" font-size="11" font-family="{MONO}" text-anchor="middle">{int(rate*100)}%</text>')
    s.append(f'<text x="{ml+(W-ml-mr)/2:.1f}" y="{H-24}" fill="{FAINT}" font-size="11" text-anchor="middle">invisible-character insertion rate</text>')
    s.append(f'<text x="18" y="{mt+(H-mt-mb)/2:.1f}" fill="{FAINT}" font-size="11" text-anchor="middle" transform="rotate(-90 18 {mt+(H-mt-mb)/2:.1f})">context length</text>')
    # legend
    lx = ml + (W - ml - mr) + 14
    s.append(f'<text x="{lx}" y="{mt+8}" fill="{FAINT}" font-size="10" font-family="{MONO}">attack</text>')
    s.append(f'<text x="{lx}" y="{mt+20}" fill="{FAINT}" font-size="10" font-family="{MONO}">success</text>')
    for k, f in enumerate([1.0, 0.5, 0.0]):
        yy = mt + 34 + k * 20
        s.append(f'<rect x="{lx}" y="{yy}" width="14" height="14" fill="{ramp(f)}" rx="2" stroke="{RULE}"/>')
        lab = {1.0: "8/8", 0.5: "4/8", 0.0: "0/8"}[f]
        s.append(f'<text x="{lx+20}" y="{yy+11}" fill="{SOFT}" font-size="10" font-family="{MONO}">{lab}</text>')
    s.append('</svg>')
    return "".join(s)


PAGE = """<title>Watermark vs Context</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin:0; background:{BG}; color:{INK}; font-family:{FONT}; }}
.wrap {{ max-width:760px; margin:0 auto; padding:56px 28px 80px; }}
h1 {{ font-size:30px; letter-spacing:-0.02em; margin:0 0 10px; font-weight:700; }}
.sub {{ color:{SOFT}; font-size:16px; line-height:1.55; margin:0 0 40px; max-width:60ch; }}
.card {{ border:1px solid {RULE}; border-radius:8px; background:#0a0a0a; padding:22px; margin:26px 0; }}
.card h2 {{ font-size:18px; margin:0 0 4px; font-weight:700; letter-spacing:-0.01em; }}
.card p {{ color:{SOFT}; font-size:14px; line-height:1.6; margin:0 0 18px; max-width:64ch; }}
.card svg {{ width:100%; height:auto; }}
.foot {{ color:{FAINT}; font-family:{MONO}; font-size:12px; margin-top:36px; }}
.foot a {{ color:{SOFT}; }}
</style>
<div class="wrap">
  <h1>The detector wins by waiting</h1>
  <p class="sub">Two views of the same mechanism, from the study data. The watermark accumulates statistical confidence as text gets longer, so an attacker has to insert proportionally more to keep the signal scrambled. The attack is not context-agnostic; its cost rises with length.</p>

  <div class="card">
    <h2>1 &middot; Watermark strength rises with context</h2>
    <p>gpt-oss-120b, one 32k generation per domain scored at 2k / 8k / 32k prefixes. The detector's z-score grows roughly like &radic;n while per-token signal stays flat, so longer text is <em>harder</em> to attack, not easier. Prose carries the strongest mark at every length.</p>
    {C1}
  </div>

  <div class="card">
    <h2>2 &middot; The attack has to work harder as context grows</h2>
    <p>gpt-oss-20b, 240-cell benchmark (code domain). Each cell is how many of 8 random insertion seeds drove the detector below threshold. 10% insertion clears 1k tokens but fails completely by 4k; only 30% held across everything tested. The required insertion rate climbs with length.</p>
    {C2}
  </div>

  <p class="foot">Data: <a href="https://github.com/aloshdenny/claude-awm">github.com/aloshdenny/claude-awm</a> &middot; results/res_120b_lengths.json, results/res_vs_bench_20b.json &middot; regenerate with site/gen_charts.py</p>
</div>
"""


def build():
    html = PAGE.format(BG=BG, INK=INK, SOFT=SOFT, FAINT=FAINT, RULE=RULE,
                       FONT=FONT, MONO=MONO, C1=chart1(), C2=chart2())
    open(os.path.join(HERE, "charts.html"), "w").write(html)
    # full-doc copy for GitHub Pages (viewport meta)
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           + html.split("</style>", 1)[0].replace("<title>", '<title>', 1) + "</style>\n</head>\n<body>\n"
           + html.split("</style>", 1)[1] + "\n</body>\n</html>\n")
    open(os.path.join(REPO, "docs", "charts.html"), "w").write(doc)
    print("wrote site/charts.html and docs/charts.html")


if __name__ == "__main__":
    build()
