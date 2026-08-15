"""
SynthID-Text watermarking for MLX models (gpt-oss-20b MXFP4-Q8 on Apple Silicon).

transformers cannot run gpt-oss's native MXFP4 on MPS (falls back to bf16, ~42GB)
and the CPU MXFP4 path needs Triton, which has no macOS build. So the model runs
in MLX while the watermark math -- which never touches the model -- is computed
here and bridged in through mlx_lm's logits_processors hook.

The g-value/key derivation is re-expressed as plain vectorized int64 arithmetic
(HF uses torch.vmap over the vocab, which costs ~0.5s/token at a 201k vocab).
verify_against_hf() asserts this fast path is bit-identical to HF's own
_compute_keys/sample_g_values, so the output is scorable by the stock HF detector.
"""
import os, json, time, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

MULT = 6364136223846793005          # LCG multiplier (newlib/musl), matches HF
INCR = 1
TABLE_SIZE = 2 ** 16
TABLE_SEED = 0


def torch_device():
    """Must match the device the DETECTOR is built on (synthid_robustness.DEVICE)."""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_sampling_table(size=TABLE_SIZE, seed=TABLE_SEED, device=None):
    """Reproduce HF's table exactly: torch.randint(0, 2, (size,), seed).

    torch.randint with a seeded Generator produces DIFFERENT values per device
    -- a CPU table and an MPS table agree only ~50% of the time, i.e. not at
    all. The table must therefore be built on the same device the detector
    uses, or generation and detection silently use unrelated g-values.
    """
    import torch
    dev = device or torch_device()
    g = torch.Generator(device=dev).manual_seed(seed)
    t = torch.randint(low=0, high=2, size=(size,), generator=g, device=dev)
    return t.cpu().numpy().astype(np.int64)


class SynthIDMLX:
    """Applies SynthID tournament reweighting to MLX logits, one step at a time."""

    def __init__(self, ngram_len, keys, vocab_size):
        self.ngram_len = ngram_len
        self.keys = np.asarray(keys, dtype=np.int64)
        self.depth = len(keys)
        self.table = make_sampling_table()
        # sized lazily from the real logits width -- gpt-oss pads its output
        # layer wider than tokenizer.vocab_size, so trusting the tokenizer here
        # yields a short g array and a shape mismatch on the first token.
        self.vocab = np.arange(vocab_size, dtype=np.int64) if vocab_size else None
        self.calls = 0

    def _ensure_vocab(self, n):
        if self.vocab is None or self.vocab.shape[0] != n:
            self.vocab = np.arange(n, dtype=np.int64)

    # ---- hashing (int64 wraparound, same semantics as torch.long) ----------
    @staticmethod
    def _acc(h, data):
        for i in range(data.shape[-1]):
            h = (h + data[..., i]) * MULT + INCR
        return h

    def _context_hash(self, ctx):
        """ctx: (ngram_len-1,) int64 -> scalar hash"""
        h = np.int64(1)
        for t in ctx:
            h = np.int64(h + np.int64(t)) * np.int64(MULT) + np.int64(INCR)
        return h

    def g_values_for_vocab(self, ctx):
        """g values for every vocab token at every depth: (vocab, depth) in {0,1}."""
        with np.errstate(over="ignore"):
            h1 = self._context_hash(ctx)
            h2 = (h1 + self.vocab) * MULT + INCR             # (vocab,)
            h3 = (h2[:, None] + self.keys[None, :]) * MULT + INCR   # (vocab, depth)
        return self.table[np.mod(h3, TABLE_SIZE)]

    # ---- the tournament, as HF's update_scores does it ---------------------
    @staticmethod
    def reweight(probs, g):
        """probs (vocab,), g (vocab, depth) -> watermarked probs.

        float32 + BLAS dot for the g-mass, and an in-place update. The naive
        form promotes to float64 and allocates two 201k temporaries per layer,
        which dominated the per-token bridge cost.
        """
        p = np.ascontiguousarray(probs, dtype=np.float32)
        # (depth, vocab) so each layer slice is contiguous -> real BLAS dot
        gf = np.ascontiguousarray(g.T, dtype=np.float32)
        for gi in gf:
            m = float(gi @ p)                 # g-mass at this depth
            p *= (1.0 + gi - m)
        return p

    def __call__(self, tokens, logits):
        """mlx_lm logits_processor hook: (tokens, logits) -> logits."""
        self.calls += 1
        n = tokens.shape[0] if tokens.ndim == 1 else tokens.shape[-1]
        if n < self.ngram_len - 1:
            return logits

        flat = logits.reshape(-1)
        lg = np.array(flat.astype(mx.float32), copy=True)
        ctx = np.array(tokens[-(self.ngram_len - 1):], copy=True).astype(np.int64)

        self._ensure_vocab(lg.shape[0])
        g = self.g_values_for_vocab(ctx)
        lg = lg - lg.max()
        p = np.exp(lg)
        p /= p.sum()
        p = self.reweight(p, g)
        out = np.log(np.maximum(p, 1e-38)).astype(np.float32)
        return mx.array(out).reshape(logits.shape)


def verify_against_hf(ngram_len, keys, vocab_size=4096, trials=3):
    """Assert the fast arithmetic path matches HF's vmap implementation exactly."""
    import torch
    from transformers import SynthIDTextWatermarkLogitsProcessor

    # MUST build on the same device the detector uses -- validating on CPU
    # while the detector runs on MPS hides a total table mismatch.
    dev = torch.device(torch_device())
    hf = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=ngram_len, keys=keys, sampling_table_size=TABLE_SIZE,
        sampling_table_seed=TABLE_SEED, context_history_size=1024,
        device=dev)
    mine = SynthIDMLX(ngram_len, keys, vocab_size)

    # table must match first
    assert np.array_equal(mine.table, hf.sampling_table.cpu().numpy().astype(np.int64)), \
        f"sampling table mismatch (device={dev})"

    rng = np.random.default_rng(0)
    for _ in range(trials):
        ctx = rng.integers(0, vocab_size, size=(ngram_len - 1,)).astype(np.int64)
        idx = torch.arange(vocab_size, device=dev)[None, :]
        nk, _ = hf._compute_keys(torch.tensor(ctx, device=dev)[None, :], idx)
        g_hf = hf.sample_g_values(nk)[0].cpu().numpy().astype(np.int64)  # (vocab, depth)
        g_me = mine.g_values_for_vocab(ctx)
        assert g_hf.shape == g_me.shape, f"shape {g_hf.shape} vs {g_me.shape}"
        assert np.array_equal(g_hf, g_me), "g-value mismatch vs HF"

    # and the reweighting must match update_scores
    sc = torch.tensor(rng.normal(size=(1, vocab_size)).astype(np.float32), device=dev)
    ctx = rng.integers(0, vocab_size, size=(ngram_len - 1,)).astype(np.int64)
    nk, _ = hf._compute_keys(torch.tensor(ctx, device=dev)[None, :],
                             torch.arange(vocab_size, device=dev)[None, :])
    g_hf = hf.sample_g_values(nk)
    ref = hf.update_scores(sc, g_hf.float())[0].cpu().numpy()
    p = torch.softmax(sc, dim=1)[0].cpu().numpy().astype(np.float64)
    mineout = np.log(np.maximum(mine.reweight(p, mine.g_values_for_vocab(ctx)), 1e-38))
    # compare as distributions (both are log-probs up to normalisation)
    a = np.exp(ref - ref.max()); a /= a.sum()
    b = np.exp(mineout - mineout.max()); b /= b.sum()
    err = float(np.abs(a - b).max())
    assert err < 1e-6, f"reweight mismatch, max abs prob diff {err}"
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gpt-oss-20b-MXFP4-Q8")
    ap.add_argument("--ngram-len", type=int, default=5)
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--target", type=int, default=8192)
    ap.add_argument("--ndocs", type=int, default=1)
    ap.add_argument("--resp", type=int, default=512)
    ap.add_argument("--out", default="docs_20b.json")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    import random
    rng = random.Random(0)
    keys = [rng.randrange(1, 2**16) for _ in range(a.depth)]

    print("verifying fast path against HF reference...", flush=True)
    err = verify_against_hf(a.ngram_len, keys)
    print(f"  OK: g-values bit-identical, reweight max prob diff {err:.2e}", flush=True)
    json.dump(keys, open("keys_20b.json", "w"))
    if a.verify_only:
        return

    # MLX retains freed buffers in a cache by default. On a 17GB box with a
    # 12GB model that retention is the difference between running and
    # deadlocking in swap, so cap it and keep the wired set bounded.
    cache_mb = int(os.environ.get("MLX_CACHE_MB", 384))
    mx.set_cache_limit(cache_mb * 1024 * 1024)
    print(f"mlx cache limit: {cache_mb} MB", flush=True)

    print(f"loading {a.model} ...", flush=True)
    t0 = time.time()
    model, tok = load(a.model)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    proc = SynthIDMLX(a.ngram_len, keys, None)   # sized from real logits width
    eos_ids = set(getattr(tok, "eos_token_ids", None) or
                  [i for i in [getattr(tok, "eos_token_id", None)] if i is not None])
    print(f"  eos ids: {sorted(eos_ids)}", flush=True)
    sampler = make_sampler(temp=1.0, top_p=1.0)

    PROMPTS = [
        "Explain how a suspension bridge distributes load.",
        "Describe the water cycle in detail.",
        "Write about the development of the printing press.",
        "Explain how vaccines train the immune system.",
        "Describe how coral reefs form and why they matter.",
        "Explain the basics of plate tectonics.",
        "Write about the history of standardized timekeeping.",
        "Describe how a refrigerator moves heat.",
    ]

    docs = []
    if os.path.exists(a.out):
        docs = json.load(open(a.out))
        print(f"  resumed {len(docs)} docs from {a.out}", flush=True)

    t0 = time.time()
    while len(docs) < a.ndocs:
        stream, i = [], len(docs) * 13
        while len(stream) < a.target:
            p = PROMPTS[i % len(PROMPTS)]; i += 1
            msgs = [{"role": "user", "content": p}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
            got = []
            for res in generate_step(mx.array(ids), model, max_tokens=a.resp,
                                     sampler=sampler, logits_processors=[proc]):
                tokid = int(res[0] if isinstance(res, tuple) else res.token)
                if tokid in eos_ids:
                    break
                got.append(tokid)
            stream.extend(got)
            mx.clear_cache()          # return buffers between responses
            el = time.time() - t0
            print(f"  doc{len(docs)}: {len(stream):6d}/{a.target} tok "
                  f"({len(stream)/max(el,1e-9):.2f} tok/s, {el/60:.1f} min, "
                  f"mem {mx.get_active_memory()/1e9:.1f}GB)", flush=True)
            json.dump(docs + [stream], open(a.out, "w"))
        docs.append(stream[:a.target])
        json.dump(docs, open(a.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
