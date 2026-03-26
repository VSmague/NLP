"""
rerun_cls_only.py
==================
Reruns ONLY the classification task with per-example logging,
and merges the results back into the existing cache.

Does NOT recompute QA or Translation.
Requires the existing downstream_results_layer10.pkl cache.
"""

import os
import json
import pickle
import string
import configparser
import collections
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score

# ── Auth ──────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("secrets.ini")
os.environ["HF_TOKEN"] = config["huggingface"]["token"]

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen3-0.6B"
SAE_RELEASE    = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR       = "sae_features"
ABLATION_LAYER = 10
CACHE_PATH     = os.path.join(SAVE_DIR, f"downstream_results_layer{ABLATION_LAYER}.pkl")
N_SAMPLES      = 200

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
CLASS_LANGS      = ['en', 'es', 'fr', 'ja', 'zh']
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'zh': 'Chinese',
}

ABLATION_CONFIGS = ["baseline", "top-1+2"]

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Load existing cache ───────────────────────────────────────────────────────
print(f"\nLoading existing cache from {CACHE_PATH}...")
with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)
results = cache["results"]
print(f"  Keys already in cache: {list(results.keys())}")

# ── Load feature indices ──────────────────────────────────────────────────────
top_index_all = torch.load(
    os.path.join(SAVE_DIR, f"layer_{ABLATION_LAYER}_indices.pt"),
    weights_only=True
)

# ── Ablation hook ─────────────────────────────────────────────────────────────
def make_ablation_hook(sae, feature_indices):
    W_dec     = sae.W_dec.T.to(device)
    dirs      = W_dec[:, feature_indices]
    norms     = torch.norm(dirs, dim=0) ** 2
    dirs_norm = dirs / norms.clamp(min=1e-8)
    def _hook(module, inp, output):
        act = (output[0] if isinstance(output, tuple) else output).to(torch.float32)
        act = act - (act @ dirs_norm) @ dirs_norm.T
        return (act.to(output[0].dtype),) + output[1:] if isinstance(output, tuple) \
               else act.to(output.dtype)
    return _hook

SENTIMENT_PROMPT = (
    "Rate the sentiment of this review on a scale from 1 (very negative) "
    "to 5 (very positive). Reply with only the number.\n\nReview: {text}\nRating:"
)

def run_classification_with_examples(model, tokenizer, samples, handle=None):
    correct, examples = 0, []
    for text, label in samples:
        prompt = SENTIMENT_PROMPT.format(text=text[:400])
        inp    = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=512).to(device)
        out    = model.generate(
            **inp, max_new_tokens=3, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        pred_str = tokenizer.decode(
            out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip()
        try:
            pred = int(pred_str[0])
            hit  = int(pred == label + 1)
        except (ValueError, IndexError):
            pred = None
            hit  = 0
        correct += hit
        examples.append({
            "text"      : text[:200],
            "gold_label": label + 1,   # 1-5
            "pred_label": pred,         # 1-5 or None
            "correct"   : hit,
        })

    acc = correct / len(samples) * 100

    # QWK — clamp predictions to valid range [1,5], replace None with 3 (neutral)
    golds = [e["gold_label"] for e in examples]
    preds_clamped = [
        max(1, min(5, e["pred_label"])) if e["pred_label"] is not None else 3
        for e in examples
    ]
    qwk = cohen_kappa_score(golds, preds_clamped, weights="quadratic",
                             labels=[1, 2, 3, 4, 5])

    return acc, qwk, examples

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()
print("  Model loaded.")

# ── Load classification data ──────────────────────────────────────────────────
print("\nLoading Amazon Reviews (Parquet)...")
PARQUET_URL = (
    "https://huggingface.co/datasets/mteb/amazon_reviews_multi/"
    "resolve/refs%2Fconvert%2Fparquet/{lang}/test/0000.parquet"
)
cls_data = {}
N_PER_CLASS = N_SAMPLES // 5   # 5 classes (labels 0-4), e.g. 200//5 = 40 each

for lang in CLASS_LANGS:
    url = PARQUET_URL.format(lang=lang)
    try:
        ds = load_dataset("parquet", data_files={"test": url}, split="test")

        # Stratified sampling: N_PER_CLASS examples per label (0-4)
        samples = []
        for label_val in range(5):
            subset = ds.filter(lambda x, lv=label_val: x["label"] == lv)
            subset = subset.shuffle(seed=42)
            for row in subset.select(range(min(N_PER_CLASS, len(subset)))):
                samples.append((row["text"], int(row["label"])))

        # Shuffle the combined stratified sample so languages aren't in label order
        import random
        random.seed(42)
        random.shuffle(samples)

        cls_data[lang] = samples
        label_dist = {k: sum(1 for _, l in samples if l == k) for k in range(5)}
        print(f"  {lang}: {len(samples)} samples  label_dist={label_dist}")
    except Exception as e:
        print(f"  {lang}: FAILED — {e}")

# ── Load SAE ──────────────────────────────────────────────────────────────────
print(f"\nLoading SAE for layer {ABLATION_LAYER}...")
sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{ABLATION_LAYER}").to(device)

# ── Run classification with example logging ───────────────────────────────────
print("\nRunning classification with example logging...")
results.setdefault("cls_examples", {})

for lang in CLASS_LANGS:
    if lang not in cls_data:
        continue
    j       = TARGET_LANGUAGES.index(lang)
    samples = cls_data[lang]
    results["cls_examples"].setdefault(lang, {})

    for cfg in ABLATION_CONFIGS:
        print(f"  {LANG_NAMES.get(lang, lang):12s} | {cfg} ...", end=" ", flush=True)

        if cfg == "baseline":
            handle = None
        else:
            feat   = top_index_all[j, :2].to(device)
            hook   = make_ablation_hook(sae, feat)
            handle = model.model.layers[ABLATION_LAYER].register_forward_hook(hook)

        acc, qwk, exs = run_classification_with_examples(model, tokenizer, samples)
        results["cls_examples"][lang][cfg] = exs

        # Store scalar results
        results["classification"][lang][cfg] = acc
        results.setdefault("classification_qwk", {}).setdefault(lang, {})[cfg] = qwk

        if handle is not None:
            handle.remove()

        # Print distribution
        preds   = [e["pred_label"] for e in exs]
        preds_c = [p for p in preds if p is not None]
        n_none  = preds.count(None)
        dist    = dict(sorted(collections.Counter(preds_c).items()))
        if n_none: dist["None"] = n_none
        n_uniq  = len(set(preds_c))
        flag    = " ← CONSTANT" if n_uniq == 1 and n_none == 0 else ""
        print(f"acc={acc:.1f}%  qwk={qwk:+.3f}  dist={dist}  unique={n_uniq}{flag}")

del sae

# ── Save back to cache ────────────────────────────────────────────────────────
cache["results"] = results
with open(CACHE_PATH, "wb") as f:
    pickle.dump(cache, f)
print(f"\n✅ Cache updated with cls_examples → {CACHE_PATH}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n── Summary: prediction distributions ──")
for lang in CLASS_LANGS:
    if lang not in results.get("cls_examples", {}):
        continue
    print(f"\n{LANG_NAMES.get(lang, lang)}")
    gold_labels = [e["gold_label"] for e in results["cls_examples"][lang]["baseline"]]
    gold_dist   = dict(sorted(collections.Counter(gold_labels).items()))
    majority    = max(collections.Counter(gold_labels), key=collections.Counter(gold_labels).get)
    maj_acc     = collections.Counter(gold_labels)[majority] / len(gold_labels) * 100
    print(f"  gold distribution : {gold_dist}  → majority class={majority} ({maj_acc:.1f}%)")

    for cfg in ABLATION_CONFIGS:
        exs   = results["cls_examples"][lang].get(cfg, [])
        preds = [e["pred_label"] for e in exs]
        preds_clean = [p for p in preds if p is not None]
        dist  = dict(sorted(collections.Counter(preds_clean).items()))
        n_none = preds.count(None)
        if n_none: dist["None"] = n_none
        n_uniq = len(set(preds_clean))
        acc   = results["classification"][lang].get(cfg, 0)
        qwk   = results.get("classification_qwk", {}).get(lang, {}).get(cfg, float("nan"))
        flag  = " ← CONSTANT PREDICTOR" if n_uniq == 1 and n_none == 0 else ""
        print(f"  {cfg:12s}       : acc={acc:.1f}%  qwk={qwk:+.3f}  dist={dist}{flag}")