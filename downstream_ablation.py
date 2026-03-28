"""
downstream_ablation.py
=======================
Evaluates the impact of ablating top-2 language-specific SAE features
on 2 downstream multilingual tasks:

  1. QA             — XQuAD (en/es/ar/th/vi/zh) — F1 + EM
  2. Classification  — Amazon Reviews (en/es/fr/ja/zh) — Accuracy + QWK
                       Stratified: 40 examples per class (labels 0-4)

Requires: sae_features/ (from compute_nu_scores.py) + HF_TOKEN in secrets.ini
"""

import os, re, json, pickle, string, configparser, collections
import numpy as np, random, torch, torch.nn.functional as F
import matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from datasets import load_dataset

config = configparser.ConfigParser()
config.read("secrets.ini")
os.environ["HF_TOKEN"] = config["huggingface"]["token"]

MODEL_ID       = "Qwen/Qwen3-0.6B"
SAE_RELEASE    = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR       = "sae_features"
OUTPUT_DIR     = "output/plots_downstream"
ABLATION_LAYER = 10
CACHE_PATH     = os.path.join(SAVE_DIR, f"downstream_results_layer{ABLATION_LAYER}.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECOMPUTE    = False
N_SAMPLES    = 200   # QA samples per language
N_PER_CLASS  = 40    # Classification: per class (5 classes => 200 total)

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese',
    'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic',
}

QA_LANGS = ['en', 'es', 'ar', 'th', 'vi', 'zh']
XQUAD_LANG_MAP = {
    'en': 'xquad.en', 'es': 'xquad.es', 'ar': 'xquad.ar',
    'th': 'xquad.th', 'vi': 'xquad.vi', 'zh': 'xquad.zh',
}

CLASS_LANGS = ['en', 'es', 'fr', 'ja', 'zh']
PARQUET_URL = (
    "https://huggingface.co/datasets/mteb/amazon_reviews_multi/"
    "resolve/refs%2Fconvert%2Fparquet/{lang}/test/0000.parquet"
)

ABLATION_CONFIGS = ["baseline", "top-1+2"]

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load indices
print("\n[0] Loading SAE feature indices...")
with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)
LAYERS = meta["layers"]
assert ABLATION_LAYER in LAYERS
top_index_all = torch.load(
    os.path.join(SAVE_DIR, f"layer_{ABLATION_LAYER}_indices.pt"), weights_only=True)
print(f"   Layer {ABLATION_LAYER}  shape={tuple(top_index_all.shape)}")


# Ablation hook
def make_ablation_hook(sae, feature_indices):
    W_dec     = sae.W_dec.T.to(device)
    dirs      = W_dec[:, feature_indices]
    norms     = torch.norm(dirs, dim=0) ** 2
    dirs_norm = dirs / norms.clamp(min=1e-8)
    def _hook(module, inp, output):
        act = (output[0] if isinstance(output, tuple) else output).to(torch.float32)
        act = act - (act @ dirs_norm) @ dirs_norm.T
        if isinstance(output, tuple):
            return (act.to(output[0].dtype),) + output[1:]
        return act.to(output.dtype)
    return _hook

def register_ablation(model, hook):
    return model.model.layers[ABLATION_LAYER].register_forward_hook(hook)


# QA
def normalize_answer(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def f1_score(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    n = sum(common.values())
    if n == 0: return 0.0
    p = n / len(pred_toks); r = n / len(gold_toks)
    return 2 * p * r / (p + r)

def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))

def build_qa_prompt(ctx, q):
    return (f"Answer the question based on the context. Give only the answer.\n\n"
            f"Context: {ctx}\nQuestion: {q}\nAnswer:")

def run_qa(model, tokenizer, samples):
    f1s, ems, examples = [], [], []
    for ctx, q, gold in samples:
        inp = tokenizer(build_qa_prompt(ctx, q), return_tensors='pt',
                        truncation=True, max_length=512).to(device)
        out = model.generate(**inp, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(out[0][inp['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip().split('\n')[0]
        f1 = f1_score(pred, gold); em = exact_match(pred, gold)
        f1s.append(f1); ems.append(em)
        examples.append({"context": ctx[:300], "question": q, "gold": gold,
                         "pred": pred, "f1": round(f1*100,1), "em": int(em)})
    return float(np.mean(f1s))*100, float(np.mean(ems))*100, examples


# Classification — stratified, Acc + QWK
SENTIMENT_PROMPT = (
    "Rate the sentiment of this review on a scale from 1 (very negative) "
    "to 5 (very positive). Reply with only the number.\n\nReview: {text}\nRating:"
)

def run_classification(model, tokenizer, samples):
    correct, examples = 0, []
    for text, label in samples:
        inp = tokenizer(SENTIMENT_PROMPT.format(text=text[:400]),
                        return_tensors='pt', truncation=True, max_length=512).to(device)
        out = model.generate(**inp, max_new_tokens=3, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        pred_str = tokenizer.decode(out[0][inp['input_ids'].shape[1]:],
                                    skip_special_tokens=True).strip()
        try:
            pred = int(pred_str[0]); hit = int(pred == label + 1)
        except (ValueError, IndexError):
            pred = None; hit = 0
        correct += hit
        examples.append({"text": text[:200], "gold_label": label+1,
                         "pred_label": pred, "correct": hit})
    acc = correct / len(samples) * 100
    golds = [e["gold_label"] for e in examples]
    preds_c = [max(1, min(5, e["pred_label"])) if e["pred_label"] is not None else 3
               for e in examples]
    qwk = cohen_kappa_score(golds, preds_c, weights="quadratic", labels=[1,2,3,4,5])
    return acc, qwk, examples


# Data loading
def load_qa_data():
    print("   Loading XQuAD...")
    data = {}
    for lang in QA_LANGS:
        ds = load_dataset("google/xquad", XQUAD_LANG_MAP[lang], split="validation")
        data[lang] = [(r['context'], r['question'], r['answers']['text'][0])
                      for r in ds.select(range(min(N_SAMPLES, len(ds))))]
        print(f"     {lang}: {len(data[lang])} samples")
    return data

def load_classification_data():
    print(f"   Loading Amazon Reviews (stratified {N_PER_CLASS}/class)...")
    data = {}
    for lang in CLASS_LANGS:
        url = PARQUET_URL.format(lang=lang)
        try:
            ds = load_dataset("parquet", data_files={"test": url}, split="test")
            samples = []
            for lv in range(5):
                subset = ds.filter(lambda x, lv=lv: x["label"] == lv).shuffle(seed=42)
                for row in subset.select(range(min(N_PER_CLASS, len(subset)))):
                    samples.append((row["text"], int(row["label"])))
            random.seed(42); random.shuffle(samples)
            data[lang] = samples
            dist = {k: sum(1 for _, l in samples if l == k) for k in range(5)}
            print(f"     {lang}: {len(samples)} samples  dist={dist}")
        except Exception as e:
            print(f"     {lang}: FAILED — {e}")
    return data


# Main loop
if RECOMPUTE or not os.path.exists(CACHE_PATH):
    import time as _time

    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval(); print("   Model loaded.")

    print("\n[2] Loading datasets...")
    qa_data  = load_qa_data()
    cls_data = load_classification_data()

    print(f"\n[3] Loading SAE for layer {ABLATION_LAYER}...")
    sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{ABLATION_LAYER}").to(device)

    if os.path.exists(CACHE_PATH) and not RECOMPUTE:
        with open(CACHE_PATH, "rb") as _f:
            results = pickle.load(_f).get("results", {})
        print("   Resuming from partial cache")
    else:
        results = {}

    for _k, _langs in [('qa', QA_LANGS), ('qa_em', QA_LANGS),
                        ('classification', CLASS_LANGS),
                        ('classification_qwk', CLASS_LANGS)]:
        results.setdefault(_k, {})
        for _l in _langs: results[_k].setdefault(_l, {})

    def _save():
        with open(CACHE_PATH, "wb") as _f:
            pickle.dump({"results": results, "ablation_layer": ABLATION_LAYER,
                         "n_samples": N_SAMPLES, "n_per_class": N_PER_CLASS}, _f)

    def _fmt(t):
        m, s = divmod(int(t), 60); h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

    total = (len(QA_LANGS) + len(CLASS_LANGS)) * len(ABLATION_CONFIGS)
    done  = [0]; run_times = []; t0_global = _time.time()

    def _prog(task, lang, cfg, metric, secs):
        done[0] += 1; run_times.append(secs)
        avg = sum(run_times)/len(run_times)
        eta = avg * (total - done[0])
        pct = done[0]/total*100
        print(f"     [{done[0]:2d}/{total}] {pct:4.0f}%  {task} | {LANG_NAMES[lang]:12s} | "
              f"{cfg:10s} | {metric}  (run {_fmt(secs)}, avg {avg:.0f}s, ETA {_fmt(eta)})",
              flush=True)

    # QA
    print(f"\n[4] QA (XQuAD) — {len(QA_LANGS)} langs × {len(ABLATION_CONFIGS)} configs")
    for lang in QA_LANGS:
        j = TARGET_LANGUAGES.index(lang)
        for cfg in ABLATION_CONFIGS:
            if cfg in results['qa'].get(lang, {}):
                print(f"     [SKIP] QA | {LANG_NAMES[lang]} | {cfg}"); done[0] += 1; continue
            avg_s = f"{sum(run_times)/len(run_times):.0f}s/run" if run_times else "warming up"
            print(f"     → QA | {LANG_NAMES[lang]:12s} | {cfg}  [{done[0]+1}/{total}]  avg={avg_s}", flush=True)
            t0 = _time.time()
            handle = None
            if cfg != "baseline":
                handle = register_ablation(model, make_ablation_hook(sae, top_index_all[j, :2].to(device)))
            f1, em, exs = run_qa(model, tokenizer, qa_data[lang])
            results['qa'   ][lang][cfg] = f1
            results['qa_em'][lang][cfg] = em
            results.setdefault('qa_examples', {}).setdefault(lang, {})[cfg] = exs
            _save()
            if handle: handle.remove()
            _prog("QA  ", lang, cfg, f"F1={f1:5.1f}  EM={em:5.1f}", _time.time()-t0)

    # Classification
    print(f"\n[5] Classification (Amazon Reviews) — {len(CLASS_LANGS)} langs × {len(ABLATION_CONFIGS)} configs")
    for lang in CLASS_LANGS:
        j = TARGET_LANGUAGES.index(lang)
        for cfg in ABLATION_CONFIGS:
            if cfg in results['classification'].get(lang, {}):
                print(f"     [SKIP] CLS | {LANG_NAMES[lang]} | {cfg}"); done[0] += 1; continue
            avg_s = f"{sum(run_times)/len(run_times):.0f}s/run" if run_times else "warming up"
            print(f"     → CLS | {LANG_NAMES[lang]:12s} | {cfg}  [{done[0]+1}/{total}]  avg={avg_s}", flush=True)
            t0 = _time.time()
            handle = None
            if cfg != "baseline":
                handle = register_ablation(model, make_ablation_hook(sae, top_index_all[j, :2].to(device)))
            acc, qwk, exs = run_classification(model, tokenizer, cls_data[lang])
            results['classification'    ][lang][cfg] = acc
            results['classification_qwk'][lang][cfg] = qwk
            results.setdefault('cls_examples', {}).setdefault(lang, {})[cfg] = exs
            _save()
            if handle: handle.remove()
            _prog("CLS ", lang, cfg, f"Acc={acc:5.1f}%  QWK={qwk:+.3f}", _time.time()-t0)

    del sae; _save()
    print(f"\nSaved to {CACHE_PATH}")

else:
    print(f"\nLoading from cache ({CACHE_PATH})...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    results = cache['results']
    ABLATION_LAYER = cache['ablation_layer']
    N_SAMPLES = cache['n_samples']
    print(f"   layer={ABLATION_LAYER}  n_samples={N_SAMPLES}")


# Plots
CFG_COLORS  = {"baseline": "#888888", "top-1+2": "#C44E52"}
CFG_HATCHES = {"baseline": "",        "top-1+2": "//"}
CFG_LABELS  = {"baseline": "Baseline", "top-1+2": "Ablate top-1 & top-2"}

def bar_group(ax, langs, task_res, ylabel, title, hline=None):
    x = np.arange(len(langs)); w = 0.38
    for k, cfg in enumerate(ABLATION_CONFIGS):
        vals = [task_res.get(l, {}).get(cfg, np.nan) for l in langs]
        ax.bar(x + (k-0.5)*w, vals, w, label=CFG_LABELS[cfg],
               color=CFG_COLORS[cfg], hatch=CFG_HATCHES[cfg], alpha=0.85, edgecolor='white')
    if hline is not None:
        ax.axhline(hline, color='grey', linestyle='--', linewidth=0.8, label=f'Chance ({hline})')
    ax.set_xticks(x); ax.set_xticklabels([LANG_NAMES[l] for l in langs], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

def delta_bars(ax, langs, task_res, ylabel, title):
    x = np.arange(len(langs))
    deltas = [task_res.get(l,{}).get("top-1+2", np.nan) - task_res.get(l,{}).get("baseline", np.nan)
              for l in langs]
    bars = ax.bar(x, deltas, color=['#C44E52' if d < 0 else '#55A868' for d in deltas],
                  alpha=0.85, edgecolor='white', width=0.6)
    for bar, d in zip(bars, deltas):
        if not np.isnan(d):
            ax.text(bar.get_x()+bar.get_width()/2, d, f"{d:+.1f}",
                    ha='center', va='bottom' if d >= 0 else 'top', fontsize=8, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels([LANG_NAMES[l] for l in langs], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f"Δ {ylabel}", fontsize=10); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Plot 1: QA absolute
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bar_group(axes[0], QA_LANGS, results['qa'],    "F1",  "QA — F1 (XQuAD)")
bar_group(axes[1], QA_LANGS, results['qa_em'], "EM",  "QA — Exact Match")
fig.suptitle(f"QA — baseline vs ablation (layer {ABLATION_LAYER})", fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "1_qa_absolute.png"), dpi=150); plt.close()
print("→ 1_qa_absolute.png")

# Plot 2: QA delta
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
delta_bars(axes[0], QA_LANGS, results['qa'],    "F1", "ΔF1 — QA")
delta_bars(axes[1], QA_LANGS, results['qa_em'], "EM", "ΔEM — QA")
fig.suptitle("QA performance drop after ablation", fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "2_qa_delta.png"), dpi=150); plt.close()
print("→ 2_qa_delta.png")

# Plot 3: Classification absolute
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bar_group(axes[0], CLASS_LANGS, results['classification'],     "Accuracy (%)", "Classification — Accuracy", hline=20)
bar_group(axes[1], CLASS_LANGS, results['classification_qwk'], "QWK",          "Classification — QWK",      hline=0)
fig.suptitle(f"Classification — baseline vs ablation (layer {ABLATION_LAYER}, stratified 40/class)",
             fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "3_cls_absolute.png"), dpi=150); plt.close()
print("→ 3_cls_absolute.png")

# Plot 4: Classification delta
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
delta_bars(axes[0], CLASS_LANGS, results['classification'],     "Accuracy (%)", "ΔAccuracy — Classification")
delta_bars(axes[1], CLASS_LANGS, results['classification_qwk'], "QWK",          "ΔQWK — Classification")
fig.suptitle("Classification performance drop after ablation", fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "4_cls_delta.png"), dpi=150); plt.close()
print("→ 4_cls_delta.png")

# CSV
rows = []
for task_key, langs, metric in [
    ('qa', QA_LANGS, 'F1'), ('qa_em', QA_LANGS, 'EM'),
    ('classification', CLASS_LANGS, 'Accuracy'),
    ('classification_qwk', CLASS_LANGS, 'QWK'),
]:
    for lang in langs:
        base = results.get(task_key, {}).get(lang, {}).get('baseline', np.nan)
        abl  = results.get(task_key, {}).get(lang, {}).get('top-1+2', np.nan)
        rows.append({'task': task_key, 'metric': metric, 'lang': lang,
                     'baseline': round(base,3), 'ablated': round(abl,3), 'delta': round(abl-base,3)})
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "downstream_results.csv"), index=False)
print("→ downstream_results.csv")
print(f"\nAll outputs in '{OUTPUT_DIR}/'  —  4 plots + 1 CSV")