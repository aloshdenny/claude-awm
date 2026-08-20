"""
Metal-resident SynthID processor.

The numpy processor in synthid_mlx.py computes g-values for the whole vocab on
the CPU every token. At gpt-oss's ~201k vocab x depth 30 that is ~6M int64 ops
per token on one core, and it dominates the per-token cost -- the same CPU-bound
stall that made FAST_WM=1 the wrong choice on an H100.

MLX int64 multiply wraps exactly like numpy/torch int64 (verified), so the LCG
hash chain ports to Metal unchanged and stays bit-identical. verify_matches()
asserts that against the existing numpy path before we trust it.
"""
import numpy as np
import mlx.core as mx

MULT = 6364136223846793005
INCR = 1
TABLE_SIZE = 2 ** 16


class SynthIDMetal:
    def __init__(self, ngram_len, keys, table_np):
        self.ngram_len = ngram_len
        self.keys = mx.array(np.asarray(keys, dtype=np.int64), dtype=mx.int64)
        self.depth = len(keys)
        self.table = mx.array(table_np.astype(np.int64), dtype=mx.int64)
        self.vocab = None
        self.calls = 0

    def _ensure_vocab(self, n):
        if self.vocab is None or self.vocab.shape[0] != n:
            self.vocab = mx.arange(n, dtype=mx.int64)

    def _context_hash(self, ctx):
        h = np.int64(1)
        with np.errstate(over="ignore"):
            for t in ctx:
                h = np.int64(h + np.int64(t)) * np.int64(MULT) + np.int64(INCR)
        return int(h)

    def g_values_for_vocab(self, ctx):
        """(vocab, depth) g-values, computed on the GPU."""
        h1 = self._context_hash(ctx)
        h2 = (self.vocab + h1) * MULT + INCR                    # (vocab,)
        h3 = (h2[:, None] + self.keys[None, :]) * MULT + INCR   # (vocab, depth)
        idx = mx.remainder(h3, TABLE_SIZE)
        # remainder of a negative int64 is negative in both numpy and mlx; the
        # numpy path indexes with np.mod (always non-negative), so match it.
        idx = mx.where(idx < 0, idx + TABLE_SIZE, idx)
        return self.table[idx]

    @staticmethod
    def reweight(probs, g):
        """probs (vocab,) float32, g (vocab, depth) -> watermarked probs."""
        p = probs.astype(mx.float32)
        gf = g.astype(mx.float32).T                # (depth, vocab)
        for i in range(gf.shape[0]):
            gi = gf[i]
            m = mx.sum(gi * p)
            p = p * (1.0 + gi - m)
        return p

    def __call__(self, tokens, logits):
        self.calls += 1
        n = tokens.shape[0] if tokens.ndim == 1 else tokens.shape[-1]
        if n < self.ngram_len - 1:
            return logits
        flat = logits.reshape(-1)
        self._ensure_vocab(flat.shape[0])
        ctx = np.array(tokens[-(self.ngram_len - 1):], copy=True).astype(np.int64)
        g = self.g_values_for_vocab(ctx)
        lg = flat.astype(mx.float32)
        lg = lg - mx.max(lg)
        p = mx.exp(lg)
        p = p / mx.sum(p)
        p = self.reweight(p, g)
        out = mx.log(mx.maximum(p, 1e-38))
        return out.reshape(logits.shape)


def verify_matches(ngram_len, keys, table_np, vocab_size=8192, trials=3):
    """Assert the Metal path is bit-identical to the numpy path it replaces."""
    import sys
    sys.path.insert(0, "/tmp/claude-awm/src")
    from synthid_mlx import SynthIDMLX

    cpu = SynthIDMLX(ngram_len, keys, vocab_size)
    cpu.table = table_np.astype(np.int64)
    gpu = SynthIDMetal(ngram_len, keys, table_np)
    gpu._ensure_vocab(vocab_size)

    rng = np.random.default_rng(0)
    for _ in range(trials):
        ctx = rng.integers(0, vocab_size, size=(ngram_len - 1,)).astype(np.int64)
        g_cpu = cpu.g_values_for_vocab(ctx)
        g_gpu = np.array(gpu.g_values_for_vocab(ctx))
        assert g_cpu.shape == g_gpu.shape, f"{g_cpu.shape} vs {g_gpu.shape}"
        assert np.array_equal(g_cpu, g_gpu), "g-value mismatch CPU vs Metal"

    # reweight must agree as a distribution.
    # NB: the numpy reweight mutates its input in place (`p *= ...`, and
    # ascontiguousarray is a no-op on an already-float32 array), so each path
    # gets its own copy -- otherwise the second call reads the first's output.
    logits = rng.normal(size=(vocab_size,)).astype(np.float32)
    ctx = rng.integers(0, vocab_size, size=(ngram_len - 1,)).astype(np.int64)
    p0 = np.exp(logits - logits.max()); p0 /= p0.sum(); p0 = p0.astype(np.float32)
    a = cpu.reweight(p0.copy(), cpu.g_values_for_vocab(ctx)); a = a / a.sum()
    b = np.array(gpu.reweight(mx.array(p0.copy()), gpu.g_values_for_vocab(ctx))); b = b / b.sum()
    err = float(np.abs(a - b).max())
    assert err < 1e-6, f"reweight mismatch, max prob diff {err}"
    return err
