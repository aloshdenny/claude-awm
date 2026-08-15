"""
SynthID-Text robustness sweep on Qwen3.5-4B.

Benchmark 1: watermarked text, detection vs. context length (1k -> 32k)
Benchmark 2: same text after hardcoded cosmetic edits, same sweep

Null distribution comes from scoring the SAME text under independent WRONG key
sets. That controls for content exactly and needs no extra generation, so the
z-scores are apples-to-apples across every condition.
"""
import os, re, sys, json, time, difflib, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    SynthIDTextWatermarkingConfig, SynthIDTextWatermarkLogitsProcessor,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

MULT, INCR = 6364136223846793005, 1


class FastSynthID(LogitsProcessor):
    """Same watermark as HF's processor, computed with vectorised int64 numpy.

    HF's _compute_keys vmaps over the full vocabulary, materialising
    [batch, vocab, depth] int64 tensors on MPS that the caching allocator never
    returns -- that ballooned the 4B run to 14GB and deadlocked it in swap.
    The LCG factorises into two broadcast multiplies, so this is the identical
    computation at a fraction of the memory. validate() asserts bit-equality.
    """

    def __init__(self, ngram_len, keys, table):
        self.ngram_len = ngram_len
        self.keys = np.asarray(keys, dtype=np.int64)
        self.table = table
        self.vocab = None

    def g_values(self, ctx):
        with np.errstate(over="ignore"):
            h = np.int64(1)
            for t in ctx:
                h = np.int64(h + np.int64(t)) * np.int64(MULT) + np.int64(INCR)
            h2 = (h + self.vocab) * MULT + INCR
            h3 = (h2[:, None] + self.keys[None, :]) * MULT + INCR
        return self.table[np.mod(h3, self.table.shape[0])]

    @staticmethod
    def reweight(p, g):
        p = np.ascontiguousarray(p, dtype=np.float32)
        gf = np.ascontiguousarray(g.T, dtype=np.float32)
        for gi in gf:
            p = p * (1.0 + gi - float(gi @ p))
        return p

    def __call__(self, input_ids, scores):
        B, V = scores.shape
        if self.vocab is None or self.vocab.shape[0] != V:
            self.vocab = np.arange(V, dtype=np.int64)
        if input_ids.shape[1] < self.ngram_len - 1:
            return scores
        ctx = input_ids[:, -(self.ngram_len - 1):].detach().cpu().numpy().astype(np.int64)
        sc = scores.detach().float().cpu().numpy()
        out = np.empty_like(sc)
        for b in range(B):
            lg = sc[b] - sc[b].max()
            p = np.exp(lg); p /= p.sum()
            p = self.reweight(p, self.g_values(ctx[b]))
            out[b] = np.log(np.maximum(p, 1e-38))
        return torch.from_numpy(out).to(scores.device, scores.dtype)


def validate_fast(ngram_len, keys, hf_proc, vocab=4096):
    """Assert FastSynthID reproduces HF's g-values exactly."""
    fast = FastSynthID(ngram_len, keys, hf_proc.sampling_table.cpu().numpy().astype(np.int64))
    fast.vocab = np.arange(vocab, dtype=np.int64)
    rng = np.random.default_rng(0)
    for _ in range(3):
        ctx = rng.integers(0, vocab, size=(ngram_len - 1,)).astype(np.int64)
        nk, _ = hf_proc._compute_keys(torch.tensor(ctx)[None, :].to(hf_proc.device),
                                      torch.arange(vocab, device=hf_proc.device)[None, :])
        g_hf = hf_proc.sample_g_values(nk)[0].cpu().numpy().astype(np.int64)
        assert np.array_equal(g_hf, fast.g_values(ctx)), "FastSynthID != HF g-values"
    return True

MODEL     = os.environ.get("MODEL", "Qwen/Qwen3.5-4B")
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
NGRAM_LEN = 5
DEPTH     = 30

# production defaults: OpenAI / Anthropic APIs both default to temperature=1.0,
# top_p=1.0. No truncation, which is also what the watermark wants.
TEMPERATURE, TOP_P, TOP_K = 1.0, 1.0, 0

LENGTHS   = [int(x) for x in os.environ.get(
    "LENGTHS", "1024,2048,4096,8192,16384,32768").split(",")]
TARGET    = max(LENGTHS)
N_DOCS    = int(os.environ.get("N_DOCS", 3))
RESP_TOK  = 1024
BATCH     = int(os.environ.get("BATCH", 4))
N_NULL    = 16          # wrong-key replicates -> empirical null
OUT       = os.environ.get("OUT", "robustness_results.json")

rng = random.Random(0)
TRUE_KEYS  = [rng.randrange(1, 2**16) for _ in range(DEPTH)]
NULL_KEYS  = [[rng.randrange(1, 2**16) for _ in range(DEPTH)] for _ in range(N_NULL)]

PROMPTS = [
    "Explain how a suspension bridge distributes load.",
    "Describe the water cycle in detail.",
    "Write about the development of the printing press.",
    "Explain how vaccines train the immune system.",
    "Describe how coral reefs form and why they matter.",
    "Explain the basics of plate tectonics.",
    "Write about the history of standardized timekeeping.",
    "Describe how a refrigerator moves heat.",
    "Explain why leaves change colour in autumn.",
    "Describe the archaeology of Pompeii.",
    "Explain how GPS satellites correct for relativity.",
    "Write about the domestication of wheat.",
    "Describe how noise-cancelling headphones work.",
    "Explain the carbon cycle and its reservoirs.",
    "Write about the construction of the Panama Canal.",
    "Describe how migratory birds navigate.",
]

# ---------------------------------------------------------------- attacks ---
def a_none(t):    return t
def a_dash(t):    return t.replace("—", "-").replace("–", "-")
def a_amp(t):     return re.sub(r"\band\b", "&", t)
def a_quotes(t):
    for a, b in [("“",'"'),("”",'"'),("‘","'"),("’","'"),("…","...")]:
        t = t.replace(a, b)
    return t
def a_space(t):   return t.replace(" ", " ").replace("  ", " ")
def a_all(t):     return a_space(a_quotes(a_amp(a_dash(t))))

def a_contract(t):
    for a, b in [("do not","don't"),("cannot","can't"),("it is","it's"),
                 ("that is","that's"),("is not","isn't"),("does not","doesn't"),
                 ("will not","won't"),("they are","they're")]:
        t = re.sub(rf"\b{a}\b", b, t)
    return t

SYNS = {"very":"extremely","large":"big","big":"large","use":"utilize",
        "many":"numerous","begin":"start","show":"demonstrate","help":"assist",
        "make":"create","also":"additionally","however":"but","because":"since",
        "important":"crucial","different":"distinct","about":"regarding"}
def a_synonym(t):
    return re.sub(r"\b(" + "|".join(SYNS) + r")\b", lambda m: SYNS[m.group(0)], t)

def _word_delete(t, p, seed=1234):
    r = random.Random(seed)
    w = t.split(" ")
    return " ".join(x for x in w if r.random() >= p)

def a_del(p):
    """Delete p of whitespace words -> clean dose-response knob."""
    return lambda t: _word_delete(t, p)

# ---- markdown ----
def a_md_partial(t):
    """Strip emphasis only: **bold**, *italic*, `code`."""
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    return t

def a_md_full(t):
    """Strip all markdown structure: headers, emphasis, lists, links, fences."""
    t = a_md_partial(t)
    t = re.sub(r"^```.*$", "", t, flags=re.M)          # fences
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)  # headers
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)       # bullets
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.M)       # numbered
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)           # quotes
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)       # links
    t = re.sub(r"^\s*([-*_]\s*){3,}$", "", t, flags=re.M)  # rules
    return t

# ---- whitespace ----
def a_ws_strip(t):
    """Collapse every whitespace run to a single space."""
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]{2,}", " ", t)).strip()

def a_ws_add(t, seed=7):
    """Inject a stray space after ~8% of words."""
    r = random.Random(seed)
    return "".join(w + ("  " if r.random() < 0.08 else "") for w in re.split(r"(\s)", t))

def a_ws_tabs(t):
    """Leading indentation -> tabs, newlines doubled."""
    t = re.sub(r"^ {2,}", "\t", t, flags=re.M)
    return t.replace("\n", "\n\n")

# ---- invisible characters ----------------------------------------------
# Zero-width and other non-printing chars. These desync BPE merges without
# any visible change, so they "survive" reading and copy-paste -- but a
# normalizing detector strips them trivially (see NORMALIZE below).
ZW = ["​", "‌", "‍", "⁠", "﻿"]     # zero-width family
def a_zwsp(t, p=0.10, seed=21):
    """Insert a zero-width space between characters with probability p."""
    r = random.Random(seed)
    out = []
    for ch in t:
        out.append(ch)
        if ch != " " and r.random() < p:
            out.append(r.choice(ZW))
    return "".join(out)

def a_zw_word(t, p=0.5, seed=22):
    """Insert a zero-width char after ~p of word boundaries only."""
    r = random.Random(seed)
    return re.sub(r"(\w+)", lambda m: m.group(1) + (r.choice(ZW) if r.random() < p else ""), t)

def a_nbsp(t, p=0.15, seed=23):
    """Replace some normal spaces with non-breaking spaces (visually identical)."""
    r = random.Random(seed)
    return "".join(" " if (c == " " and r.random() < p) else c for c in t)

def a_homoglyph(t, p=0.06, seed=24):
    """Swap a few ASCII letters for Cyrillic/Greek lookalikes."""
    HG = {"a":"а","e":"е","o":"о","p":"р","c":"с",
          "y":"у","x":"х","i":"і"}
    r = random.Random(seed)
    return "".join(HG[c] if (c in HG and r.random() < p) else c for c in t)

# ---- bidi override injection ----
# RLO/LRO/PDF and isolates. Visually reorders text; injected between tokens
# they desync the byte stream. NFKC does NOT strip these -> a normalizer must
# handle bidi controls explicitly (this is the interesting defensive gap).
BIDI = ["‪", "‫", "‬", "‭", "‮", "⁦", "⁧", "⁨", "⁩"]
def a_bidi(t, p=0.08, seed=26):
    """Insert balanced bidi override/isolate pairs around ~p of words."""
    r = random.Random(seed)
    def wrap(m):
        if r.random() >= p:
            return m.group(0)
        return "‭" + m.group(0) + "‬"   # LRO ... PDF (visually a no-op)
    return re.sub(r"\w+", wrap, t)

# ---- whitespace-shift steganography ----
def a_lineshift(t, p=0.3, seed=27):
    """Append trailing spaces to ~p of lines (classic line-shift stego channel)."""
    r = random.Random(seed)
    return "\n".join(ln + ("   " if r.random() < p else "") for ln in t.split("\n"))

def a_wordshift(t, p=0.15, seed=28):
    """Widen ~p of inter-word gaps to double space (word-shift stego)."""
    r = random.Random(seed)
    return re.sub(r"(?<=\w) (?=\w)", lambda m: "  " if r.random() < p else " ", t)

# ---- code-specific ----
def a_code_indent(t, p=0.3, seed=25):
    """Add an extra tab to ~p of already-indented lines (code attack)."""
    r = random.Random(seed)
    out = []
    for line in t.split("\n"):
        if line[:1] in (" ", "\t") and r.random() < p:
            out.append("\t" + line)
        else:
            out.append(line)
    return "\n".join(out)

# ---- THE DEFENSE: input normalization ----------------------------------
# zero-width + bidi controls + other format (Cf) chars
_FMT_RE = re.compile("[​‌‍‎‏‪‫‬‭‮"
                     "⁠⁡⁢⁣⁤⁦⁧⁨⁩﻿]")
def normalize(t):
    """What a competent detector runs before scoring untrusted text.
    Strips zero-width + bidi control chars, NFKC-folds nbsp/compat homoglyphs,
    collapses whitespace runs and trailing whitespace."""
    import unicodedata
    t = _FMT_RE.sub("", t)
    t = unicodedata.normalize("NFKC", t)          # folds nbsp->space, compat forms
    t = re.sub(r"[ \t]+", " ", t)                 # collapse ALL multi-space (stego channel)
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)        # strip trailing/leading line whitespace
    return t

# ---- dialect / abbreviation substitution dictionary --------------------
# Dispersed lexical rewrites: these fire wherever the word occurs, which is
# scattered rather than clustered -- the geometry that hurt the detector most.
BRIT = {
    "color":"colour","colors":"colours","colored":"coloured","behavior":"behaviour",
    "behaviors":"behaviours","favorite":"favourite","honor":"honour","labor":"labour",
    "neighbor":"neighbour","harbor":"harbour","vapor":"vapour","odor":"odour",
    "center":"centre","centers":"centres","meter":"metre","meters":"metres",
    "liter":"litre","liters":"litres","fiber":"fibre","fibers":"fibres",
    "theater":"theatre","defense":"defence","offense":"offence","license":"licence",
    "practice":"practise","gray":"grey","aluminum":"aluminium","program":"programme",
    "catalog":"catalogue","dialog":"dialogue","analog":"analogue","plow":"plough",
    "mold":"mould","smolder":"smoulder","draft":"draught","tire":"tyre",
    "curb":"kerb","jail":"gaol","math":"maths","airplane":"aeroplane",
    "toward":"towards","traveled":"travelled","traveling":"travelling",
    "modeling":"modelling","canceled":"cancelled","labeled":"labelled",
    "fueled":"fuelled","signaling":"signalling","marvelous":"marvellous",
}
ABBREV = {
    "for example":"e.g.","that is":"i.e.","and so on":"etc.","approximately":"approx.",
    "versus":"vs.","information":"info","application":"app","laboratory":"lab",
    "temperature":"temp.","maximum":"max.","minimum":"min.","department":"dept.",
    "government":"govt.","estimated":"est.","approximately equal":"~",
    "percent":"%","number":"no.","figure":"fig.","equation":"eq.","section":"sec.",
    "including":"incl.","without":"w/o","with":"w/","because":"b/c",
}
_IZE = re.compile(r"\b(\w{3,}?)(iz)(e|es|ed|ing|ation|ations)\b")
_YZE = re.compile(r"\b(\w{2,}?)(yz)(e|es|ed|ing)\b")


def _sub_dict(t, table, p, seed):
    """Apply each eligible substitution independently with probability p."""
    r = random.Random(seed)
    keys = sorted(table, key=len, reverse=True)
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b",
                     re.IGNORECASE)

    def rep(m):
        if r.random() >= p:
            return m.group(0)
        src = m.group(0)
        out = table[src.lower()]
        return out.capitalize() if src[:1].isupper() else out
    return pat.sub(rep, t)


def a_brit(t, p=1.0, seed=11):
    t = _sub_dict(t, BRIT, p, seed)
    r = random.Random(seed + 1)
    t = _IZE.sub(lambda m: m.group(1) + ("is" if r.random() < p else "iz") + m.group(3), t)
    t = _YZE.sub(lambda m: m.group(1) + ("ys" if r.random() < p else "yz") + m.group(3), t)
    return t


def a_abbrev(t, p=1.0, seed=12):
    return _sub_dict(t, ABBREV, p, seed)


def a_dict(p):
    """Randomised dialect+abbreviation rewrite at dispersion rate p."""
    return lambda t: a_abbrev(a_brit(t, p), p)


def a_dict_ws(p):
    """Dictionary rewrite PLUS scattered whitespace -- max dispersion."""
    return lambda t: a_ws_add(a_abbrev(a_brit(t, p), p), seed=13)


def a_mix(t):
    return a_ws_strip(a_md_full(a_all(t)))

def a_mix_heavy(t):
    return a_ws_add(a_ws_strip(a_md_full(a_synonym(a_contract(a_all(t))))))

ATTACKS = [
    ("roundtrip", a_none),     # control: decode -> re-encode only, no edits
    # --- markdown / whitespace ---
    ("md_part",   a_md_partial),
    ("md_full",   a_md_full),
    ("ws_strip",  a_ws_strip),
    ("ws_add",    a_ws_add),
    ("ws_tabs",   a_ws_tabs),
    ("mix",       a_mix),
    ("mix_heavy", a_mix_heavy),
    # --- the hardcoded cosmetic edits ---
    ("dash",      a_dash),
    ("amp",       a_amp),
    ("quotes",    a_quotes),
    ("space",     a_space),
    ("contract",  a_contract),
    ("synonym",   a_synonym),
    ("cosmetic",  a_all),
    # --- dose-response: random word deletion at increasing rate ---
    ("del_01",    a_del(0.01)),
    ("del_02",    a_del(0.02)),
    ("del_05",    a_del(0.05)),
    ("del_10",    a_del(0.10)),
    ("del_20",    a_del(0.20)),
    ("del_40",    a_del(0.40)),
    # --- randomised dialect/abbreviation dictionary (dispersion sweep) ---
    ("brit",      a_brit),
    ("abbrev",    a_abbrev),
    ("dict_25",   a_dict(0.25)),
    ("dict_50",   a_dict(0.50)),
    ("dict_100",  a_dict(1.00)),
    ("dictws_50", a_dict_ws(0.50)),
    ("dictws_100", a_dict_ws(1.00)),
    # --- everything, maximum dispersion ---
    ("kitchen",   lambda t: a_ws_add(a_synonym(a_contract(a_abbrev(a_brit(a_all(t))))), seed=17)),
]

# Focused tokenizer-desync + invisible-char family for the DEFENSE experiment.
DESYNC_ATTACKS = [
    ("roundtrip", a_none),
    ("zwsp_10",   lambda t: a_zwsp(t, 0.10)),   # zero-width every ~10% chars
    ("zwsp_30",   lambda t: a_zwsp(t, 0.30)),
    ("zw_word",   lambda t: a_zw_word(t, 0.5)), # zero-width at word ends
    ("nbsp",      a_nbsp),                       # non-breaking spaces
    ("bidi",      a_bidi),                       # RLO/LRO/PDF balanced pairs
    ("homoglyph", a_homoglyph),                  # Cyrillic/Greek lookalikes
    ("lineshift", a_lineshift),                  # trailing-space stego
    ("wordshift", a_wordshift),                  # inter-word double-space stego
    ("combo",     lambda t: a_zwsp(a_nbsp(a_bidi(t)), 0.15)),
]

# ws_add and code_indent are visible edits (they change layout a reader sees),
# so they're excluded from the "invisible stego" set above. Keep them available:
VISIBLE_EDITS = [("ws_add", a_ws_add), ("code_indent", lambda t: a_code_indent(t, 0.3))]
if os.environ.get("ATTACK_SET") == "desync":
    ATTACKS = DESYNC_ATTACKS

# ------------------------------------------------------------- generation ---
def log(*a):
    print(*a, flush=True)

def build_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    return tok, model

def render(tok, p):
    m = [{"role": "user", "content": p}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def gen_batch(tok, model, prompts, cfg):
    enc = tok([render(tok, p) for p in prompts], return_tensors="pt", padding=True).to(DEVICE)
    if hasattr(cfg, "state"):
        cfg.state = None          # HF processor is stateful; stale ctx across batches
    out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE,
                         top_p=TOP_P, top_k=TOP_K, max_new_tokens=RESP_TOK,
                         pad_token_id=tok.pad_token_id,
                         logits_processor=LogitsProcessorList([cfg]))
    gen = out[:, enc["input_ids"].shape[1]:]
    res = []
    for row in gen:
        row = row[row != tok.pad_token_id]
        res.append(row.tolist())
    return res

DOCS_FILE = os.environ.get("DOCS", "docs.json")

def build_docs(tok, model):
    """Concatenate independent watermarked responses into long transcripts.

    Checkpoints to DOCS_FILE after every batch so a long run stays usable
    even if it is interrupted partway.
    """
    ref = make_proc(TRUE_KEYS)
    validate_fast(NGRAM_LEN, TRUE_KEYS, ref)
    log("  FastSynthID validated bit-identical to HF g-values")
    if os.environ.get("FAST_WM", "1") == "1":
        cfg = FastSynthID(NGRAM_LEN, TRUE_KEYS,
                          ref.sampling_table.cpu().numpy().astype(np.int64))
        log("  using FastSynthID (numpy, low-memory)")
    else:
        # On a GPU with headroom, HF's own vectorised processor is far faster
        # than the CPU numpy path (which would bottleneck the GPU at ~15 tok/s).
        cfg = ref
        log("  using HF SynthIDTextWatermarkLogitsProcessor (GPU)")
    del ref
    docs, t0, done = [], time.time(), 0
    if os.path.exists(DOCS_FILE):
        docs = json.load(open(DOCS_FILE))
        done = sum(len(d) for d in docs)
        log(f"  resumed {len(docs)} docs ({done} tok) from {DOCS_FILE}")
    while len(docs) < N_DOCS:
        stream, i = [], len(docs) * 977
        while len(stream) < TARGET:
            batch = [PROMPTS[(i + k) % len(PROMPTS)] for k in range(BATCH)]
            i += BATCH
            for r in gen_batch(tok, model, batch, cfg):
                stream.extend(r)
            if DEVICE == "mps":
                torch.mps.empty_cache()   # MPS allocator does not return freed blocks
            el = time.time() - t0
            gen = len(stream) + sum(len(d) for d in docs) - done
            log(f"  doc{len(docs)}: {len(stream):6d}/{TARGET} tok  "
                f"({gen/max(el,1e-9):.1f} tok/s, {el/60:.1f} min elapsed)")
            json.dump(docs + [stream], open(DOCS_FILE, "w"))
        docs.append(stream[:TARGET])
        json.dump(docs, open(DOCS_FILE, "w"))
    return docs[:N_DOCS]

# -------------------------------------------------------------- detection ---
# Canonical sampling table: built once with a CPU generator so it is
# DEVICE-INDEPENDENT. HF seeds torch.randint on the compute device, and
# torch.randint gives different values per device (MPS/CUDA/CPU tables agree
# only ~50%). That makes docs generated on one device unscorable on another.
# Overwriting every processor's table with this canonical CPU-seeded copy makes
# generation and detection portable across machines -- required for the study.
_CANON_TABLE = None
def _canonical_table(device):
    global _CANON_TABLE
    if _CANON_TABLE is None:
        g = torch.Generator(device="cpu").manual_seed(0)
        _CANON_TABLE = torch.randint(0, 2, (2**16,), generator=g)
    return _CANON_TABLE.to(device)

def make_proc(keys):
    p = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=NGRAM_LEN, keys=keys, sampling_table_size=2**16,
        sampling_table_seed=0, context_history_size=1024, device=DEVICE)
    p.sampling_table = _canonical_table(p.sampling_table.device)  # force portable table
    return p

@torch.no_grad()
def mean_g(proc, ids, eos_id):
    t = torch.tensor([ids], device=DEVICE)
    g = proc.compute_g_values(t)
    ctx = proc.compute_context_repetition_mask(t)
    eos = proc.compute_eos_token_mask(t, eos_id)[:, NGRAM_LEN - 1:]
    m = (ctx * eos).float()
    if m.sum() < 1:
        return float("nan"), 0
    s = (g.float().mean(-1) * m).sum() / m.sum()
    return float(s), int(m.sum())

def token_edit_rate(a, b):
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    same = sum(bl.size for bl in sm.get_matching_blocks())
    return 1.0 - same / max(len(a), 1)

# ------------------------------------------------------------------- main ---
def main():
    score_only = os.environ.get("SCORE_ONLY") == "1"
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id
    log(f"model={MODEL} device={DEVICE} depth={DEPTH} ngram_len={NGRAM_LEN} "
        f"temp={TEMPERATURE} top_p={TOP_P}")
    log(f"lengths={LENGTHS} docs={N_DOCS} null_keysets={N_NULL}\n")

    if score_only:
        docs = [d for d in json.load(open(DOCS_FILE)) if len(d) >= min(LENGTHS)]
        log(f"== scoring {len(docs)} cached docs "
            f"(lengths available: {[len(d) for d in docs]}) ==")
        LENGTHS[:] = [L for L in LENGTHS if L <= max(len(d) for d in docs)]
    else:
        log("== generating watermarked transcripts ==")
        tok2, model = build_model()
        docs = build_docs(tok2, model)
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    true_proc = make_proc(TRUE_KEYS)
    null_procs = [make_proc(k) for k in NULL_KEYS]

    # When DEFENSE=1, score every attack twice: raw text, and after the
    # detector's normalize() preprocessor. The gap = how much normalization
    # protects the detector.
    DEFENSE = os.environ.get("DEFENSE", "0") == "1"

    def z_of(ids):
        s, _ = mean_g(true_proc, ids, eos)
        nulls = [mean_g(p, ids, eos)[0] for p in null_procs]
        mu = sum(nulls) / len(nulls)
        sd = (sum((x - mu) ** 2 for x in nulls) / (len(nulls) - 1)) ** 0.5
        return (s - mu) / max(sd, 1e-12), s

    log("\n== scoring ==" + ("  (with DEFENSE normalization)" if DEFENSE else ""))
    results = []
    for L in LENGTHS:
        for aname, afn in ATTACKS:
            zs, zn, scores, rates = [], [], [], []
            for doc in docs:
                if len(doc) < L:
                    continue
                clean_ids = doc[:L]
                text = tok.decode(clean_ids, skip_special_tokens=True)
                attacked = afn(text)
                ids = tok(attacked, add_special_tokens=False)["input_ids"]
                base = tok(text, add_special_tokens=False)["input_ids"]

                z_raw, s = z_of(ids)
                zs.append(z_raw); scores.append(s)
                rates.append(token_edit_rate(base, ids))
                if DEFENSE:
                    nids = tok(normalize(attacked), add_special_tokens=False)["input_ids"]
                    zn.append(z_of(nids)[0])
            if not zs:
                continue
            row = dict(length=L, attack=aname, n=len(zs),
                       mean_g=sum(scores)/len(scores),
                       z=sum(zs)/len(zs),
                       z_min=min(zs),
                       edit_rate=sum(rates)/len(rates))
            if DEFENSE and zn:
                row["z_norm"] = sum(zn)/len(zn)
            results.append(row)
            extra = f" z_norm={row['z_norm']:8.2f}" if DEFENSE and zn else ""
            log(f"L={L:6d} {aname:11s} z_raw={row['z']:8.2f}{extra} "
                f"edit_rate={row['edit_rate']*100:5.2f}%")
        log("")

    with open(OUT, "w") as f:
        json.dump(dict(config=dict(model=MODEL, lengths=LENGTHS, depth=DEPTH,
                                   temperature=TEMPERATURE, n_docs=N_DOCS),
                       results=results), f, indent=1)
    log(f"wrote {OUT}")

if __name__ == "__main__":
    main()
