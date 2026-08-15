"""
Semantic-fidelity check for the stego attacks: does the text a HUMAN reads stay
identical after each attack? An attack that changes the reading is not stego.

'What a human reads' = the sequence of visible base glyphs after: removing
zero-width/format/bidi controls, resolving nbsp->space, and folding homoglyphs
back to their Latin skeleton (since a reader perceives Cyrillic 'а' as 'a').
"""
import sys, unicodedata, importlib.util as iu
sys.argv = ["x"]
spec = iu.spec_from_file_location("m", "synthid_robustness.py")
m = iu.module_from_spec(spec); spec.loader.exec_module(m)

# reverse homoglyph map: what the reader THINKS each lookalike is
HG_BACK = {"а":"a","е":"e","о":"o","р":"p","с":"c","у":"y","х":"x","і":"i",
           "ѕ":"s","ԁ":"d","ո":"n","м":"m","к":"k","т":"t","в":"b","н":"h"}
INVIS = set("​‌‍‎‏‪‫‬‭‮⁠⁡⁢⁣⁤⁦⁧⁨⁩﻿")

def perceived(t):
    """The glyph sequence a human actually reads."""
    out = []
    for ch in t:
        if ch in INVIS:            # invisible -> not seen
            continue
        ch = HG_BACK.get(ch, ch)   # homoglyph -> perceived Latin
        if ch == " ":         # nbsp -> ordinary space
            ch = " "
        out.append(ch)
    s = "".join(out)
    # a reader does not distinguish runs of spaces or trailing spaces
    import re
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s).strip()
    return s

SAMPLE = ("Suspension bridges distribute loads to substantial towers and piers. "
          "The color of the cable and the behavior under load are analyzed by "
          "engineers.\n    def compute(x):\n        return x * 2\n")

print(f"{'attack':12s} {'perceived==orig':16s} {'first divergence'}")
base = perceived(SAMPLE)
for name, fn in m.DESYNC_ATTACKS + m.VISIBLE_EDITS:
    att = fn(SAMPLE)
    p = perceived(att)
    same = (p == base)
    div = ""
    if not same:
        for i,(a,b) in enumerate(zip(base, p)):
            if a != b:
                div = f"@{i}: {a!r} vs {b!r}"; break
        else:
            div = f"len {len(base)} vs {len(p)}"
    print(f"{name:12s} {'YES (invisible)' if same else 'NO -- CHANGES READ':16s} {div}")
