"""
gpt-oss-120b watermark benchmark on Modal.

Weights live in a persistent Volume, so a crashed or re-run job never
re-downloads 65GB. Generation and scoring are separate entrypoints for the
same reason -- if generation dies we can still score whatever it checkpointed.

The watermark math is the study's, unchanged: HF's own
SynthIDTextWatermarkLogitsProcessor for generation, and a mean-g / 16-null-key
z-score for detection.
"""
import modal

app = modal.App("synthid-glm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6", "transformers>=5.0", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]", "triton", "kernels", "numpy",
        "compressed-tensors>=0.15",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/weights/hf"})
)

vol = modal.Volume.from_name("synthid-glm-weights", create_if_missing=True)
results = modal.Volume.from_name("synthid-glm-results", create_if_missing=True)

MODEL = "cyankiwi/GLM-5.2-AWQ-INT4"
NGRAM_LEN, DEPTH, N_NULL = 5, 30, 16
THRESHOLD = 2.33


@app.function(image=image, volumes={"/weights": vol}, timeout=60 * 60,
              cpu=8, memory=32768)
def download():
    """Pull the model into the Volume once. Idempotent."""
    import os, time
    from huggingface_hub import snapshot_download
    t0 = time.time()
    p = snapshot_download(MODEL, max_workers=16)
    vol.commit()
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(p) for f in fs
    )
    msg = f"downloaded {total/1e9:.1f} GB in {time.time()-t0:.0f}s -> {p}"
    print(msg)
    return msg


@app.function(image=image, gpu="B300:2", volumes={"/weights": vol, "/out": results},
              timeout=60 * 60 * 3)
def generate(domain: str, target: int = 2048):
    """Generate `target` watermarked tokens for one domain, checkpointing as it
    goes so a timeout still leaves usable data."""
    import os, json, time, random, torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              SynthIDTextWatermarkLogitsProcessor, LogitsProcessorList)

    PROMPTS = {
        "prose": [
            "Explain how a suspension bridge distributes load.",
            "Describe how coral reefs form and why they matter.",
            "Write about the development of the printing press.",
        ],
        "code": [
            "Write a Python class for a thread-safe LRU cache, with docstrings.",
            "Implement Dijkstra's shortest-path algorithm in Python using a heap.",
            "Write a Python function to parse a CSV into dicts, handling quoted fields.",
        ],
        "reasoning": [
            "A farmer has 17 sheep and all but 9 run away. Work out how many remain.",
            "If a train leaves at 3:15pm at 80km/h and another at 4:00pm at 100km/h, when does the second catch the first?",
            "Is 2**61 - 1 prime? Reason it through.",
        ],
    }[domain]

    out_path = f"/out/docs_glm_{domain}.json"
    stream = json.load(open(out_path)) if os.path.exists(out_path) else []
    if len(stream) >= target:
        return f"{domain}: already have {len(stream)} tok"

    rng = random.Random(0)
    keys = [rng.randrange(1, 2**16) for _ in range(DEPTH)]

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.pad_token or tok.eos_token
    t0 = time.time()
    # device_map="auto" packed device 1 to within 370MB of full and then OOMed
    # allocating KV cache (needed 12.88GB, had 12.51GB free of 287GB). Cap each
    # card well below its 287GB so activations and KV cache have room.
    n_gpu = torch.cuda.device_count()
    cap = os.environ.get("PER_GPU_GB", "225")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", dtype="auto",
        max_memory={i: f"{cap}GiB" for i in range(n_gpu)}).eval()
    print(f"loaded in {time.time()-t0:.0f}s, "
          f"{torch.cuda.memory_allocated()/1e9:.1f}GB on GPU", flush=True)

    proc = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=NGRAM_LEN, keys=keys, sampling_table_size=2**16,
        sampling_table_seed=0, context_history_size=1024, device="cuda:0")

    eos = tok.eos_token_id
    pi, t0 = 0, time.time()
    while len(stream) < target:
        prompt = PROMPTS[pi % len(PROMPTS)]; pi += 1
        msgs = [{"role": "user", "content": prompt}]
        enc = tok(tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True),
                  return_tensors="pt").to("cuda:0")
        gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                             top_k=0, max_new_tokens=512,
                             pad_token_id=tok.pad_token_id,
                             logits_processor=LogitsProcessorList([proc]))
        new = gen[0, enc["input_ids"].shape[1]:].tolist()
        stream.extend([t for t in new if t != eos])
        json.dump(stream, open(out_path, "w"))
        results.commit()
        el = time.time() - t0
        print(f"{domain}: {len(stream)}/{target} tok "
              f"({len(stream)/max(el,1e-9):.1f} tok/s, {el/60:.1f} min)", flush=True)
    return f"{domain}: DONE {len(stream)} tok"


@app.function(image=image, gpu="B300:2", volumes={"/out": results}, timeout=60 * 60)
def score():
    """mean-g vs 16 wrong-key nulls -> z, for every domain that has tokens."""
    import json, glob, random, torch
    from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor

    rng = random.Random(0)
    KEYS = [rng.randrange(1, 2**16) for _ in range(DEPTH)]
    nrng = random.Random(9999)
    NULLS = [[nrng.randrange(1, 2**16) for _ in range(DEPTH)] for _ in range(N_NULL)]

    def mk(k):
        return SynthIDTextWatermarkLogitsProcessor(
            ngram_len=NGRAM_LEN, keys=k, sampling_table_size=2**16,
            sampling_table_seed=0, context_history_size=1024, device="cuda:0")

    @torch.no_grad()
    def mean_g(proc, ids, eos):
        t = torch.tensor([ids], device="cuda:0")
        g = proc.compute_g_values(t)
        c = proc.compute_context_repetition_mask(t)
        e = proc.compute_eos_token_mask(t, eos)[:, NGRAM_LEN - 1:]
        m = (c * e).float()
        if m.sum() < 1:
            return float("nan"), 0
        return float((g.float().mean(-1) * m).sum() / m.sum()), int(m.sum())

    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id
    out = {}
    for f in sorted(glob.glob("/out/docs_glm_*.json")):
        dom = f.split("docs_glm_")[1].replace(".json", "")
        full = json.load(open(f))
        for L in [2048, 8192, 32768]:
            if len(full) < L:
                continue
            ids = full[:L]
            gt, n = mean_g(mk(KEYS), ids, eos)
            nl = [mean_g(mk(k), ids, eos)[0] for k in NULLS]
            mu = sum(nl) / len(nl)
            sd = (sum((x - mu) ** 2 for x in nl) / (len(nl) - 1)) ** 0.5
            z = (gt - mu) / max(sd, 1e-9)
            out[f"{dom}@{L}"] = dict(tokens=L, scored=n, mean_g=gt, z=z,
                                     detected=bool(z >= THRESHOLD))
            print(f"{dom:10s} @{L:6d} scored={n:6d} "
                  f"mean_g={gt:.4f} z={z:8.2f} "
                  f"{'DETECTED' if z >= THRESHOLD else 'not detected'}", flush=True)
    json.dump(out, open("/out/res_glm.json", "w"), indent=1)
    results.commit()
    return out
