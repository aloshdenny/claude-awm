"""
Invisible Ink connector: a read-only detection tool for Claude.

Paste text as a tool call, get back a g-value heatmap and a verdict. This
reuses the EXACT detection math from src/synthid_robustness.py (same key
seeding, same canonical sampling table, same mean-g/z-score) so the numbers
here are faithful to the study, not a reimplementation.

What this can't do: detect a REAL Anthropic or Google watermark. Detection
only works against the exact key text was generated with, and nobody outside
those companies has that key. This connector ships its own research key
(seeded the same way the study's is) purely to demonstrate the mechanism.
Text watermarked with a real production key will always read as unwatermarked
here -- that's not a bug, it's the same soundness property the whole study
relies on.

This connector is detection-only. There is no tool that returns a modified
or "cleaned" version of the input text. See site/ for the (separate, clearly
labeled) attack-transform demo, which never claims a real detection score.
"""
import logging
import os
import random

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor
try:
    from mcp.server.fastmcp import FastMCP as _MCPServer  # SDK < 2.0
except ImportError:
    from mcp.server.mcpserver import MCPServer as _MCPServer  # SDK >= 2.0

mcp = _MCPServer("invisible-ink")

# ---- identical constants + key seeding to src/synthid_robustness.py -------
NGRAM_LEN = 5
DEPTH = 30
N_NULL = 16
THRESHOLD = 2.33  # z for 1% false positive rate

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

_rng = random.Random(0)
TRUE_KEYS = [_rng.randrange(1, 2**16) for _ in range(DEPTH)]
NULL_KEYS = [[_rng.randrange(1, 2**16) for _ in range(DEPTH)] for _ in range(N_NULL)]

DEFAULT_MODEL = os.environ.get("IINK_TOKENIZER", "Qwen/Qwen3.5-0.8B")
_tok_cache = {}


def get_tokenizer(model: str):
    if model not in _tok_cache:
        _tok_cache[model] = AutoTokenizer.from_pretrained(model)
    return _tok_cache[model]


# ---- identical to make_proc() / mean_g() in synthid_robustness.py --------
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
    p.sampling_table = _canonical_table(p.sampling_table.device)
    return p


_TRUE_PROC = None
_NULL_PROCS = None


def procs():
    global _TRUE_PROC, _NULL_PROCS
    if _TRUE_PROC is None:
        _TRUE_PROC = make_proc(TRUE_KEYS)
        _NULL_PROCS = [make_proc(k) for k in NULL_KEYS]
    return _TRUE_PROC, _NULL_PROCS


@torch.no_grad()
def mean_g(proc, ids, eos_id):
    t = torch.tensor([ids], device=DEVICE)
    g = proc.compute_g_values(t)
    ctx = proc.compute_context_repetition_mask(t)
    eos = proc.compute_eos_token_mask(t, eos_id)[:, NGRAM_LEN - 1:]
    mask = (ctx * eos).float()
    if mask.sum() < 1:
        return float("nan"), 0
    s = (g.float().mean(-1) * mask).sum() / mask.sum()
    return float(s), int(mask.sum())


@torch.no_grad()
def per_token_g(proc, ids):
    """g-value per token position, for the heatmap. Positions before the
    first full ngram_len-1 context have no score."""
    t = torch.tensor([ids], device=DEVICE)
    g = proc.compute_g_values(t)[0].float().mean(-1).tolist()
    pad = len(ids) - len(g)
    return [None] * pad + g


HEAT_CHARS = " ░▒▓█"


def heat_char(g, gmin, gmax):
    if g is None:
        return " "
    t = (g - gmin) / max(gmax - gmin, 1e-6)
    idx = min(len(HEAT_CHARS) - 1, max(0, round(t * (len(HEAT_CHARS) - 1))))
    return HEAT_CHARS[idx]


def score(text: str, model: str):
    tok = get_tokenizer(model)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id
    true_proc, null_procs = procs()

    g_true, n = mean_g(true_proc, ids, eos)
    if n < 1:
        return None

    nulls = [mean_g(p, ids, eos)[0] for p in null_procs]
    mu = sum(nulls) / len(nulls)
    sd = (sum((x - mu) ** 2 for x in nulls) / (len(nulls) - 1)) ** 0.5
    z = (g_true - mu) / max(sd, 1e-9)

    token_g = per_token_g(true_proc, ids)
    tokens = [tok.decode([i]) for i in ids]
    return dict(z=z, mean_g=g_true, n_scored=n, n_tokens=len(ids),
                tokens=tokens, token_g=token_g)


@mcp.tool()
def analyze_watermark(text: str, model: str = DEFAULT_MODEL) -> str:
    """Score text against this connector's research SynthID watermark key
    and return a verdict, z-score, and a per-token g-value heatmap.

    IMPORTANT: this can only detect text watermarked with THIS connector's
    own research key. It cannot tell you whether text really came from
    Claude, ChatGPT, Gemini, or any production system -- detection requires
    the exact generation-time key, and nobody outside the company that
    issued it has that key. Arbitrary pasted text (including real AI
    output) will correctly read as "not detected" here, because it wasn't
    generated with this key. This is a demonstration of the mechanism from
    the claude-awm study, not a real-world watermark checker.

    Args:
        text: the text to analyze. Needs a few dozen tokens or more for a
            stable score; very short input may come back inconclusive.
        model: tokenizer to score with (default Qwen/Qwen3.5-0.8B, matching
            the study). Detection is tokenizer-sensitive -- scoring with a
            different tokenizer than generated the text will not work even
            if the key matches.
    """
    text = text.strip()
    if not text:
        return "No text provided."

    result = score(text, model)
    if result is None:
        return ("Not enough scoreable tokens (need at least a few dozen "
                "words with no repeated 4-token context). Try a longer "
                "passage.")

    z = result["z"]
    detected = z >= THRESHOLD
    verdict = "WATERMARKED" if detected else "NOT DETECTED"

    gs = [g for g in result["token_g"] if g is not None]
    gmin, gmax = (min(gs), max(gs)) if gs else (0.0, 1.0)
    heat = "".join(heat_char(g, gmin, gmax) for g in result["token_g"])

    lines = [
        f"## {verdict}",
        f"z = {z:.2f}  (threshold {THRESHOLD}, 1% false-positive rate)",
        f"mean g = {result['mean_g']:.3f}  over {result['n_scored']}/{result['n_tokens']} scoreable tokens",
        "",
        "Per-token g-value heat (darker = stronger bias toward this key):",
        f"`{heat}`",
        "",
        ("Reads as watermarked under this connector's research key." if detected else
         "Reads as clean under this connector's research key -- expected for "
         "any text that wasn't generated with it, including real AI output "
         "from a system whose key we don't have."),
        "",
        "_Detection-only. This tool never returns a modified version of your "
        "text._",
    ]
    return "\n".join(lines)


@mcp.tool()
def explain_finding() -> str:
    """Return a short summary of the claude-awm study's headline finding:
    which attack defeats this kind of watermark and why, with real numbers."""
    return (
        "## claude-awm: the headline finding\n\n"
        "Across ~500 measured attack/length cells on three models, surface "
        "edits (markdown stripping, dialect swaps, whitespace changes) left "
        "SynthID-style watermarks intact. One attack family did not: Unicode "
        "**variation selectors** (category Mn), which desync the detector's "
        "context-hash chain and, unlike zero-width spaces or bidi controls, "
        "are not stripped by standard input normalization.\n\n"
        "| model | domain | baseline z | after attack |\n"
        "|---|---|---|---|\n"
        "| gpt-oss-20b | prose | 45.03 | 0.72 |\n"
        "| gpt-oss-20b | code | 37.24 | 0.68 |\n"
        "| Qwen3.8-27B | prose | 35.50 | -0.67 |\n\n"
        "Threshold is z = 2.33. Full methodology, every table, and every "
        "caveat: https://github.com/aloshdenny/claude-awm/blob/main/docs/FINDINGS.md\n"
        "Interactive version: https://aloshdenny.com/claude-awm/"
    )


if __name__ == "__main__":
    mcp.run()
