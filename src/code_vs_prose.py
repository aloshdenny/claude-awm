"""
Does the SynthID watermark collapse on code (low-entropy) vs prose (high-entropy)?

Generates watermarked text in both domains, logs per-token entropy at generation
time, and scores with the mean-g detector. If code z << prose z AND code entropy
<< prose entropy, that's the mechanistic result: the watermark is structurally
weak exactly where provenance claims matter most.
"""
import os, sys, json, time, argparse, math
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
    SynthIDTextWatermarkingConfig, SynthIDTextWatermarkLogitsProcessor)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
sys.path.insert(0, os.path.dirname(__file__))
from prompts_code import PROSE, CODE

NGRAM_LEN, DEPTH = 5, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import random
rng = random.Random(0)
KEYS = [rng.randrange(1, 2**16) for _ in range(DEPTH)]

def make_proc(dev=DEVICE):
    return SynthIDTextWatermarkLogitsProcessor(ngram_len=NGRAM_LEN, keys=KEYS,
        sampling_table_size=2**16, sampling_table_seed=0,
        context_history_size=1024, device=torch.device(dev))

class EntropyTap(LogitsProcessor):
    """Wraps the watermark processor; records Shannon entropy of the model's
    distribution BEFORE watermarking, per token."""
    def __init__(self, inner):
        self.inner = inner; self.H = []
    def __call__(self, input_ids, scores):
        p = torch.softmax(scores.float(), dim=-1)
        h = -(p * torch.log2(p.clamp_min(1e-12))).sum(-1)
        self.H.extend(h.tolist())
        return self.inner(input_ids, scores)

def mean_g(proc, ids, eos):
    t = torch.tensor([ids], device=DEVICE)
    g = proc.compute_g_values(t)
    ctx = proc.compute_context_repetition_mask(t)
    em = proc.compute_eos_token_mask(t, eos)[:, NGRAM_LEN-1:]
    m = (ctx*em).float()
    if m.sum() < 1: return float("nan"), 0
    return float((g.float().mean(-1)*m).sum()/m.sum()), int(m.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--resp", type=int, default=512)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", dtype="auto").eval()
    eos = tok.eos_token_id
    proc = make_proc()
    # 16 independent wrong keysets for the empirical null (built ONCE)
    null_procs = []
    nrng = random.Random(9999)
    for _ in range(16):
        k2 = [nrng.randrange(1, 2**16) for _ in range(DEPTH)]
        null_procs.append(SynthIDTextWatermarkLogitsProcessor(ngram_len=NGRAM_LEN,
            keys=k2, sampling_table_size=2**16, sampling_table_seed=0,
            context_history_size=1024, device=torch.device(DEVICE)))

    def render(p):
        try: return tok.apply_chat_template([{"role":"user","content":p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role":"user","content":p}],
            tokenize=False, add_generation_prompt=True)

    results = {}
    for domain, prompts in [("prose", PROSE), ("code", CODE)]:
        zs, ents, mg = [], [], []
        for i in range(a.reps):
            p = prompts[i % len(prompts)]
            enc = tok([render(p)], return_tensors="pt").to(DEVICE)
            wm = make_proc()            # real watermark LogitsProcessor
            tap = EntropyTap(wm)
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                top_k=0, max_new_tokens=a.resp, pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([tap]))
            gen = out[0, enc["input_ids"].shape[1]:].tolist()
            g, n = mean_g(proc, gen, eos)
            # z vs null: score under 16 INDEPENDENT wrong keysets (built once, above)
            nulls = [mean_g(pr2, gen, eos)[0] for pr2 in null_procs]
            mu = np.mean(nulls); sd = np.std(nulls, ddof=1)
            zs.append((g-mu)/max(sd,1e-9)); ents.append(np.mean(tap.H)); mg.append(g)
            print(f"  {domain} {i}: z={zs[-1]:.1f} meanH={ents[-1]:.2f} bits n={n}", flush=True)
        results[domain] = dict(z=float(np.mean(zs)), z_sd=float(np.std(zs)),
            entropy=float(np.mean(ents)), mean_g=float(np.mean(mg)),
            per_z=zs, per_H=ents)
    results["model"] = a.model
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nPROSE: z={results['prose']['z']:.1f}  H={results['prose']['entropy']:.2f} bits")
    print(f"CODE : z={results['code']['z']:.1f}  H={results['code']['entropy']:.2f} bits")
    print(f"ratio: z {results['code']['z']/results['prose']['z']:.2f}x  "
          f"H {results['code']['entropy']/results['prose']['entropy']:.2f}x")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
