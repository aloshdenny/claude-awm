import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
import importlib.util as iu
spec = iu.spec_from_file_location("m", os.path.join(REPO, "src", "synthid_robustness.py"))
sys.argv = ["x"]
m = iu.module_from_spec(spec)
spec.loader.exec_module(m)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(m.DEVICE).eval()

proc = m.make_proc(m.TRUE_KEYS)
eos = tok.eos_token_id

PROMPTS = {
    "prose": "Explain how noise-cancelling headphones work, in a couple of paragraphs.",
    "code": "Write a Python function that implements binary search, with a docstring.",
    "mixed": "Explain what memoization is, then show a short Python example using it on Fibonacci.",
}

def render(p):
    msgs = [{"role": "user", "content": p}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

out = {}
for domain, prompt in PROMPTS.items():
    enc = tok([render(prompt)], return_tensors="pt").to(m.DEVICE)
    # use the CANONICAL-table processor via logits_processor, not
    # watermarking_config= (which makes generate() build its own default,
    # device-seeded table -- the exact generation/detection table mismatch
    # bug fixed earlier tonight).
    gen_proc = m.make_proc(m.TRUE_KEYS)
    gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                         max_new_tokens=300, pad_token_id=tok.pad_token_id,
                         logits_processor=m.LogitsProcessorList([gen_proc]))
    ids = gen[0, enc["input_ids"].shape[1]:].tolist()
    ids = [t for t in ids if t != eos]
    text = tok.decode(ids, skip_special_tokens=True)

    import random
    rng = random.Random(9999)
    nulls_g = []
    for _ in range(16):
        k2 = [rng.randrange(1, 2**16) for _ in range(m.DEPTH)]
        p2 = m.make_proc(k2)
        g, n = m.mean_g(p2, ids, eos)
        nulls_g.append(g)
    mu, sd = sum(nulls_g)/len(nulls_g), (sum((x-sum(nulls_g)/len(nulls_g))**2 for x in nulls_g)/15)**0.5
    g_raw, n_raw = m.mean_g(proc, ids, eos)
    z_raw = (g_raw - mu) / max(sd, 1e-9)

    attacked_text = m.a_vs16(text, p=0.30, seed=29)
    attacked_ids = tok(attacked_text, add_special_tokens=False)["input_ids"]
    g_att, n_att = m.mean_g(proc, attacked_ids, eos)
    nulls_att = []
    for _ in range(16):
        k2 = [rng.randrange(1, 2**16) for _ in range(m.DEPTH)]
        p2 = m.make_proc(k2)
        g, n = m.mean_g(p2, attacked_ids, eos)
        nulls_att.append(g)
    mu2 = sum(nulls_att)/len(nulls_att)
    sd2 = (sum((x-mu2)**2 for x in nulls_att)/15)**0.5
    z_att = (g_att - mu2) / max(sd2, 1e-9)

    norm_text = m.normalize(attacked_text)
    norm_ids = tok(norm_text, add_special_tokens=False)["input_ids"]
    g_norm, _ = m.mean_g(proc, norm_ids, eos)
    nulls_norm = []
    for _ in range(16):
        k2 = [rng.randrange(1, 2**16) for _ in range(m.DEPTH)]
        p2 = m.make_proc(k2)
        g, n = m.mean_g(p2, norm_ids, eos)
        nulls_norm.append(g)
    mu3 = sum(nulls_norm)/len(nulls_norm)
    sd3 = (sum((x-mu3)**2 for x in nulls_norm)/15)**0.5
    z_norm_after = (g_norm - mu3) / max(sd3, 1e-9)

    out[domain] = dict(text=text, attacked_text=attacked_text, z_raw=z_raw, z_after=z_att,
                       z_after_norm=z_norm_after, edit_rate=m.token_edit_rate(ids, attacked_ids))
    print(domain, "z_raw", round(z_raw,2), "z_after", round(z_att,2), "z_after_norm", round(z_norm_after,2))

json.dump(out, open(os.path.join(HERE, "demo_samples.json"), "w"), indent=1)
print("wrote demo_samples.json")

# --- second pass: per-token g-values for BOTH clean and attacked streams ---
def token_g(txt):
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    t = torch.tensor([ids], device=m.DEVICE)
    g = proc.compute_g_values(t)[0].float().mean(-1).tolist()  # per-position mean over depth
    pad = len(ids) - len(g)  # g starts at position ngram_len-1
    spans = []
    for i, off in enumerate(offsets):
        spans.append({"start": off[0], "end": off[1], "g": (g[i - pad] if i >= pad else None)})
    return spans

for domain, prompt in PROMPTS.items():
    out[domain]["spans"] = token_g(out[domain]["text"])
    # attacked stream: real g-values after the variation selectors re-split it.
    # (These read near-null because the seed chain is desynced -- the whole point.)
    out[domain]["attacked_spans"] = token_g(out[domain]["attacked_text"])

json.dump(out, open(os.path.join(HERE, "demo_samples.json"), "w"), indent=0)
print("wrote demo_samples.json with clean + attacked spans")
