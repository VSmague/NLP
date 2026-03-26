"""
downstream_ablation.py
=======================
Evaluates the impact of ablating top-k language-specific SAE features
on 3 downstream multilingual tasks:

  1. QA            — XQuAD          (en/es/fr/ja/zh/ar) — F1 + EM
  2. Classification — Amazon Reviews (en/es/fr/ja/zh)    — Accuracy
  3. Translation    — FLORES-200     (all 10 langs → en) — BLEU

For each task × language × ablation config (baseline / top-1 / top-1+2):
  - Run inference with optional directional ablation hook
  - Compute metric
  - Report Δmetric = ablated − baseline

Requires: sae_features/ (from ablation.py) + HF_TOKEN in secrets.ini
"""

import os
import re
import json
import pickle
import string
import configparser
import collections
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from datasets import load_dataset
import sacrebleu

# ── Auth ──────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("secrets.ini")
os.environ["HF_TOKEN"] = config["huggingface"]["token"]

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-0.6B"
SAE_RELEASE = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR    = "sae_features"
OUTPUT_DIR  = "plots_downstream"
# Layer to ablate — layer 10 is the peak ΔCE layer for most languages (see Fig5)
ABLATION_LAYER = 10
CACHE_PATH  = os.path.join(SAVE_DIR, f"downstream_results_layer{ABLATION_LAYER}.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RECOMPUTE    = False
N_SAMPLES    = 200   # samples per language per task (keep low for speed)
TOP_K_ABLATE = 2     # max rank to load from indices


TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese',
    'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic',
}

# Task-specific language coverage
# XQuAD available: ar/de/el/en/es/hi/ro/ru/th/tr/vi/zh  (no fr/ja/ko/pt)
QA_LANGS        = ['en', 'es', 'ar', 'th', 'vi', 'zh']
CLASS_LANGS     = ['en', 'es', 'fr', 'ja', 'zh']          # Amazon Reviews
TRANSL_LANGS    = [l for l in TARGET_LANGUAGES if l != 'en']  # xx → en

# XQuAD language code mapping (only langs present in the dataset)
XQUAD_LANG_MAP = {
    'en': 'xquad.en', 'es': 'xquad.es', 'ar': 'xquad.ar',
    'th': 'xquad.th', 'vi': 'xquad.vi', 'zh': 'xquad.zh',
}

# FLORES-200 language code mapping
FLORES_LANG_MAP = {
    'es': 'spa_Latn', 'fr': 'fra_Latn', 'ja': 'jpn_Jpan',
    'ko': 'kor_Hang', 'pt': 'por_Latn', 'th': 'tha_Thai',
    'vi': 'vie_Latn', 'zh': 'cmn_Hans', 'ar': 'arb_Arab',
}

# Only 2 configs: no ablation vs ablating both top-1 and top-2 features
ABLATION_CONFIGS = ["baseline", "top-1+2"]

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================================================
# Load SAE feature indices
# =============================================================================

print("\n[0] Loading SAE feature indices...")
with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)

LAYERS = meta["layers"]
assert ABLATION_LAYER in LAYERS, \
    f"ABLATION_LAYER={ABLATION_LAYER} not in cached layers {LAYERS}. Change it."

top_index_all = torch.load(
    os.path.join(SAVE_DIR, f"layer_{ABLATION_LAYER}_indices.pt"),
    weights_only=True
)  # shape: (n_langs, top_k)
print(f"   Indices loaded for layer {ABLATION_LAYER}  shape={tuple(top_index_all.shape)}")


# =============================================================================
# Ablation hook
# =============================================================================

def make_ablation_hook(sae, feature_indices):
    """Returns a forward hook that directionally ablates given features."""
    W_dec     = sae.W_dec.T.to(device)                  # (d_model, n_features)
    dirs      = W_dec[:, feature_indices]                # (d_model, k)
    norms     = torch.norm(dirs, dim=0) ** 2             # (k,)
    dirs_norm = dirs / norms.clamp(min=1e-8)             # (d_model, k)

    def _hook(module, inp, output):
        act = (output[0] if isinstance(output, tuple) else output).to(torch.float32)
        act = act - (act @ dirs_norm) @ dirs_norm.T
        if isinstance(output, tuple):
            return (act.to(output[0].dtype),) + output[1:]
        return act.to(output.dtype)

    return _hook


def register_ablation(model, hook):
    return model.model.layers[ABLATION_LAYER].register_forward_hook(hook)


# =============================================================================
# QA utilities — XQuAD (extractive QA, greedy generation)
# =============================================================================

def normalize_answer(s):
    """Lower, remove punctuation/articles/extra whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def f1_score(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common    = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    n_common  = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred_toks)
    r = n_common / len(gold_toks)
    return 2 * p * r / (p + r)

def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))

def build_qa_prompt(context, question):
    return (
        f"Answer the question based on the context. "
        f"Give only the answer, no explanation.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

def run_qa(model, tokenizer, samples, handle=None):
    f1s, ems = [], []
    for ctx, q, gold in samples:
        prompt = build_qa_prompt(ctx, q)
        inp    = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=512).to(device)
        out    = model.generate(
            **inp, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        pred = tokenizer.decode(
            out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip().split('\n')[0]
        f1s.append(f1_score(pred, gold))
        ems.append(exact_match(pred, gold))
    return float(np.mean(f1s)) * 100, float(np.mean(ems)) * 100


# =============================================================================
# Classification utilities — Amazon Reviews (0-shot accuracy, 5-class)
# =============================================================================

SENTIMENT_PROMPT = (
    "Rate the sentiment of this review on a scale from 1 (very negative) "
    "to 5 (very positive). Reply with only the number.\n\nReview: {text}\nRating:"
)

def run_classification(model, tokenizer, samples, handle=None):
    correct = 0
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
            if pred == label + 1:   # dataset labels are 0-4, ratings 1-5
                correct += 1
        except (ValueError, IndexError):
            pass
    return correct / len(samples) * 100


# =============================================================================
# Translation utilities — FLORES-200 (greedy, BLEU)
# =============================================================================

TRANSL_PROMPT = (
    "Translate the following text to English. "
    "Output only the translation, nothing else.\n\n"
    "Text: {text}\nTranslation:"
)

def run_translation(model, tokenizer, samples, handle=None):
    hypotheses = []
    references = []
    for src, ref in samples:
        prompt = TRANSL_PROMPT.format(text=src)
        inp    = tokenizer(prompt, return_tensors='pt',
                           truncation=True, max_length=256).to(device)
        out    = model.generate(
            **inp, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        hyp = tokenizer.decode(
            out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip().split('\n')[0]
        hypotheses.append(hyp)
        references.append(ref)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# =============================================================================
# Data loading
# =============================================================================

def load_qa_data():
    print("   Loading XQuAD...")
    data = {}
    for lang in QA_LANGS:
        ds = load_dataset("google/xquad", XQUAD_LANG_MAP[lang],
                          split="validation")
        samples = [
            (row['context'], row['question'], row['answers']['text'][0])
            for row in ds.select(range(min(N_SAMPLES, len(ds))))
        ]
        data[lang] = samples
        print(f"     {lang}: {len(samples)} samples")
    return data

def load_classification_data():
    print("   Loading Amazon Reviews (via Parquet)...")
    data = {}
    # Load directly from auto-converted Parquet files to avoid loading script issues
    PARQUET_URL = (
        "https://huggingface.co/datasets/mteb/amazon_reviews_multi/"
        "resolve/refs%2Fconvert%2Fparquet/{lang}/test/0000.parquet"
    )
    for lang in CLASS_LANGS:
        url = PARQUET_URL.format(lang=lang)
        try:
            ds = load_dataset("parquet", data_files={"test": url}, split="test")
            samples = [
                (row['text'], int(row['label']))
                for row in ds.select(range(min(N_SAMPLES, len(ds))))
            ]
            data[lang] = samples
            print(f"     {lang}: {len(samples)} samples")
        except Exception as e:
            print(f"     {lang}: FAILED — {e}")
    return data

def load_translation_data():
    print("   Loading FLORES-101 (gsarti/flores_101, no script)...")
    # gsarti/flores_101 is a clean Parquet-only mirror of FLORES
    # config names match the FLORES-200 language codes used in the paper
    FLORES101_LANG_MAP = {
        'es': 'spa_Latn', 'fr': 'fra_Latn', 'ja': 'jpn_Jpan',
        'ko': 'kor_Hang', 'pt': 'por_Latn', 'th': 'tha_Thai',
        'vi': 'vie_Latn', 'zh': 'zho_Hans', 'ar': 'ara_Arab',
    }
    data = {}
    try:
        ds_en = load_dataset("gsarti/flores_101", "eng",
                             split="devtest", trust_remote_code=False)
        refs_en = [row['sentence'] for row in ds_en.select(range(min(N_SAMPLES, len(ds_en))))]
    except Exception as e:
        print(f"   ERROR loading English reference: {e}")
        return data

    for lang in TRANSL_LANGS:
        flores_code = FLORES101_LANG_MAP.get(lang)
        if flores_code is None:
            print(f"     {lang}: no FLORES-101 code, skipping")
            continue
        # gsarti/flores_101 uses 3-letter codes without script suffix
        # try both with and without suffix
        for code in [flores_code.split("_")[0], flores_code]:
            try:
                ds_src = load_dataset("gsarti/flores_101", code,
                                      split="devtest", trust_remote_code=False)
                srcs = [row['sentence'] for row in ds_src.select(
                    range(min(N_SAMPLES, len(ds_src))))]
                data[lang] = list(zip(srcs, refs_en[:len(srcs)]))
                print(f"     {lang} ({code}): {len(data[lang])} samples")
                break
            except Exception:
                continue
        else:
            print(f"     {lang}: FAILED — no working code found")
    return data


# =============================================================================
# Main evaluation loop
# =============================================================================

if RECOMPUTE or not os.path.exists(CACHE_PATH):
    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    print("   Model loaded.")

    print("\n[2] Loading datasets...")
    qa_data    = load_qa_data()
    cls_data   = load_classification_data()
    transl_data = load_translation_data()

    print(f"\n[3] Loading SAE for layer {ABLATION_LAYER}...")
    sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{ABLATION_LAYER}").to(device)

    # results[task][lang][config] = metric value
    # ── Incremental cache: load partial results if they exist ────────────────
    if os.path.exists(CACHE_PATH) and not RECOMPUTE:
        with open(CACHE_PATH, "rb") as _f:
            _partial = pickle.load(_f)
        results = _partial.get("results", {})
        print(f"   Resuming from partial cache — {sum(len(v) for v in results.values())} entries already saved")
    else:
        results = {}

    # Ensure all task keys exist
    for _k, _langs in [('qa', QA_LANGS), ('qa_em', QA_LANGS),
                        ('classification', CLASS_LANGS),
                        ('translation', [l for l in TRANSL_LANGS if l in transl_data])]:
        results.setdefault(_k, {})
        for _l in _langs:
            results[_k].setdefault(_l, {})

    def _save():
        with open(CACHE_PATH, "wb") as _f:
            pickle.dump({
                "results"       : results,
                "ablation_layer": ABLATION_LAYER,
                "n_samples"     : N_SAMPLES,
                "top_k"         : TOP_K_ABLATE,
            }, _f)

    # ── Timing helper ────────────────────────────────────────────────────────
    import time as _time

    def _eta(elapsed, done, total):
        if done == 0:
            return "?"
        remaining = elapsed / done * (total - done)
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

    def _elapsed_str(t):
        m, s = divmod(int(t), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

    # Each (lang, cfg) pair = 1 unit of work
    # Estimate: QA ~slow (generation), Classification ~medium, Translation ~slow
    qa_units    = len(QA_LANGS)    * len(ABLATION_CONFIGS)
    cls_units   = len(CLASS_LANGS) * len(ABLATION_CONFIGS)
    trl_units   = len([l for l in TRANSL_LANGS if l in transl_data]) * len(ABLATION_CONFIGS)
    total_units = qa_units + cls_units + trl_units
    done_units  = [0]  # mutable container to allow mutation inside nested function
    t_global    = _time.time()

    def _print_progress(task, lang, cfg, metric_str):
        done_units[0] += 1
        elapsed = _time.time() - t_global
        eta     = _eta(elapsed, done_units[0], total_units)
        pct     = done_units[0] / total_units * 100
        print(f"     [{done_units[0]:2d}/{total_units}] {pct:4.0f}%  "
              f"{task} | {LANG_NAMES[lang]:12s} | {cfg:10s} | "
              f"{metric_str}  (elapsed {_elapsed_str(elapsed)}, ETA {eta})",
              flush=True)

    # ── QA ───────────────────────────────────────────────────────────────────
    print(f"\n[4] Evaluating QA (XQuAD)...")
    print(f"     {len(QA_LANGS)} langs × {len(ABLATION_CONFIGS)} configs = {qa_units} runs")
    t_qa = _time.time()
    for lang in QA_LANGS:
        j = TARGET_LANGUAGES.index(lang)
        samples = qa_data[lang]

        for cfg in ABLATION_CONFIGS:
            t0 = _time.time()
            if cfg == "baseline":
                handle = None
            else:
                k      = 1 if cfg == "top-1" else 2
                feat   = top_index_all[j, :k].to(device)
                hook   = make_ablation_hook(sae, feat)
                handle = register_ablation(model, hook)

            if cfg in results['qa'].get(lang, {}):
                if handle is not None:
                    handle.remove()
                print(f"     [SKIP] QA | {LANG_NAMES[lang]} | {cfg} — already cached")
                done_units[0] += 1
                continue

            f1, em = run_qa(model, tokenizer, samples)
            results['qa'   ][lang][cfg] = f1
            results['qa_em'][lang][cfg] = em
            _save()

            if handle is not None:
                handle.remove()

            _print_progress("QA  ", lang, cfg, f"F1={f1:5.1f}  EM={em:5.1f}  ({_elapsed_str(_time.time()-t0)}/run)")

    print(f"   QA done in {_elapsed_str(_time.time()-t_qa)}  "
          f"— avg {(_time.time()-t_qa)/qa_units:.1f}s/run")

    # ── Classification ────────────────────────────────────────────────────────
    print(f"\n[5] Evaluating Classification (Amazon Reviews)...")
    print(f"     {len(CLASS_LANGS)} langs × {len(ABLATION_CONFIGS)} configs = {cls_units} runs")
    t_cls = _time.time()
    for lang in CLASS_LANGS:
        j = TARGET_LANGUAGES.index(lang)
        samples = cls_data[lang]

        for cfg in ABLATION_CONFIGS:
            t0 = _time.time()
            if cfg == "baseline":
                handle = None
            else:
                k      = 1 if cfg == "top-1" else 2
                feat   = top_index_all[j, :k].to(device)
                hook   = make_ablation_hook(sae, feat)
                handle = register_ablation(model, hook)

            if cfg in results['classification'].get(lang, {}):
                if handle is not None:
                    handle.remove()
                print(f"     [SKIP] CLS | {LANG_NAMES[lang]} | {cfg} — already cached")
                done_units[0] += 1
                continue

            acc = run_classification(model, tokenizer, samples)
            results['classification'][lang][cfg] = acc
            _save()

            if handle is not None:
                handle.remove()

            _print_progress("CLS ", lang, cfg, f"Acc={acc:5.1f}%  ({_elapsed_str(_time.time()-t0)}/run)")

    print(f"   Classification done in {_elapsed_str(_time.time()-t_cls)}  "
          f"— avg {(_time.time()-t_cls)/cls_units:.1f}s/run")

    # ── Translation ───────────────────────────────────────────────────────────
    transl_langs_present = [l for l in TRANSL_LANGS if l in transl_data]
    print(f"\n[6] Evaluating Translation (FLORES, xx→en)...")
    print(f"     {len(transl_langs_present)} langs × {len(ABLATION_CONFIGS)} configs = {trl_units} runs")
    t_trl = _time.time()
    for lang in transl_langs_present:
        j = TARGET_LANGUAGES.index(lang)
        samples = transl_data[lang]

        for cfg in ABLATION_CONFIGS:
            t0 = _time.time()
            if cfg == "baseline":
                handle = None
            else:
                k      = 1 if cfg == "top-1" else 2
                feat   = top_index_all[j, :k].to(device)
                hook   = make_ablation_hook(sae, feat)
                handle = register_ablation(model, hook)

            if cfg in results['translation'].get(lang, {}):
                if handle is not None:
                    handle.remove()
                print(f"     [SKIP] TRL | {LANG_NAMES[lang]} | {cfg} — already cached")
                done_units[0] += 1
                continue

            bleu = run_translation(model, tokenizer, samples)
            results['translation'][lang][cfg] = bleu
            _save()

            if handle is not None:
                handle.remove()

            _print_progress("TRL ", lang, cfg, f"BLEU={bleu:5.2f}  ({_elapsed_str(_time.time()-t0)}/run)")

    print(f"   Translation done in {_elapsed_str(_time.time()-t_trl)}  "
          f"— avg {(_time.time()-t_trl)/trl_units:.1f}s/run")

    del sae

    _save()
    print(f"\n✅ All results saved to {CACHE_PATH}")

else:
    print(f"\n[1-6] Loading results from cache ({CACHE_PATH})...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    results        = cache['results']
    ABLATION_LAYER = cache['ablation_layer']
    N_SAMPLES      = cache['n_samples']
    print(f"   ✅ layer={ABLATION_LAYER}  n_samples={N_SAMPLES}")


# =============================================================================
# PLOTS
# =============================================================================

CFG_COLORS = {
    "baseline" : "#888888",
    "top-1"    : "#4C72B0",
    "top-1+2"  : "#C44E52",
}
CFG_MARKERS = {"baseline": "o", "top-1": "s", "top-1+2": "^"}

def delta_df(task_results, configs=("top-1", "top-1+2")):
    """Return DataFrame of Δmetric = ablated − baseline per lang × config."""
    rows = []
    for lang, cfg_vals in task_results.items():
        base = cfg_vals.get("baseline", np.nan)
        for cfg in configs:
            val = cfg_vals.get(cfg, np.nan)
            rows.append({
                "lang"  : LANG_NAMES.get(lang, lang),
                "config": cfg,
                "value" : val,
                "delta" : val - base,
            })
    return pd.DataFrame(rows)


# ── Plot 1 — Absolute metrics side by side (3 tasks) ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

task_specs = [
    ("QA — F1 (XQuAD)",              results['qa'],             QA_LANGS,     "F1 score"),
    ("Classification — Acc (Amazon)", results['classification'], CLASS_LANGS,  "Accuracy (%)"),
    ("Translation — BLEU (→ EN)",     results['translation'],    TRANSL_LANGS, "BLEU"),
]

for ax, (title, task_res, langs, ylabel) in zip(axes, task_specs):
    langs_present = [l for l in langs if l in task_res]
    x     = np.arange(len(langs_present))
    width = 0.28

    for k, cfg in enumerate(ABLATION_CONFIGS):
        vals = [task_res[l].get(cfg, np.nan) for l in langs_present]
        ax.bar(x + (k - 1) * width, vals, width,
               label=cfg, color=CFG_COLORS[cfg], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs_present],
                       rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    f"Downstream performance — baseline vs ablation (layer {ABLATION_LAYER})\n"
    f"Model: Qwen3-0.6B  |  n={N_SAMPLES} samples per language",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_absolute_metrics.png"), dpi=150)
plt.close()
print("→ 1_absolute_metrics.png")


# ── Plot 2 — Δmetric heatmap (3 tasks × all languages) ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

delta_specs = [
    ("ΔF1 — QA",            results['qa'],             QA_LANGS),
    ("ΔAcc — Classification", results['classification'], CLASS_LANGS),
    ("ΔBLEU — Translation",  results['translation'],    TRANSL_LANGS),
]

for ax, (title, task_res, langs) in zip(axes, delta_specs):
    langs_p  = [l for l in langs if l in task_res]
    configs  = ["top-1", "top-1+2"]
    matrix   = np.array([
        [task_res[l].get(cfg, np.nan) - task_res[l].get("baseline", np.nan)
         for l in langs_p]
        for cfg in configs
    ])   # shape (2, n_langs)

    vmax = np.nanmax(np.abs(matrix))
    im   = ax.imshow(matrix, cmap='RdBu', aspect='auto',
                     vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(langs_p)))
    ax.set_xticklabels([LANG_NAMES[l] for l in langs_p],
                       rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')

    for i, cfg in enumerate(configs):
        for j, l in enumerate(langs_p):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(val) > vmax * 0.5 else 'black')
    plt.colorbar(im, ax=ax, label="Δ (ablated − baseline)")

fig.suptitle(
    f"Δ performance after ablation (layer {ABLATION_LAYER}) — "
    "negative = degradation caused by ablation",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_delta_heatmap.png"), dpi=150)
plt.close()
print("→ 2_delta_heatmap.png")


# ── Plot 3 — Δmetric bar chart per task, grouped by config ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (title, task_res, langs, ylabel) in zip(axes, task_specs):
    langs_p = [l for l in langs if l in task_res]
    x       = np.arange(len(langs_p))
    width   = 0.38

    for k, cfg in enumerate(["top-1", "top-1+2"]):
        deltas = [
            task_res[l].get(cfg, np.nan) - task_res[l].get("baseline", np.nan)
            for l in langs_p
        ]
        offset = (k - 0.5) * width
        bars   = ax.bar(x + offset, deltas, width,
                        label=cfg, color=CFG_COLORS[cfg], alpha=0.85,
                        edgecolor='white')
        # Colour bars red if negative
        for bar, d in zip(bars, deltas):
            if not np.isnan(d) and d < 0:
                bar.set_edgecolor('#8B0000')
                bar.set_linewidth(1.5)

    ax.axhline(0, color='black', linewidth=0.9, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs_p],
                       rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(f"Δ{ylabel}", fontsize=10)
    ax.set_title(f"Δ{title}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    f"Performance drop after ablation (layer {ABLATION_LAYER}) — "
    "negative bars = ablation hurts performance",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_delta_bar.png"), dpi=150)
plt.close()
print("→ 3_delta_bar.png")


# ── Plot 4 — EM for QA separately ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (metric_key, ylabel) in zip(axes, [('qa', 'F1'), ('qa_em', 'EM')]):
    task_res = results[metric_key]
    langs_p  = [l for l in QA_LANGS if l in task_res]
    x        = np.arange(len(langs_p))
    width    = 0.28

    for k, cfg in enumerate(ABLATION_CONFIGS):
        vals = [task_res[l].get(cfg, np.nan) for l in langs_p]
        ax.bar(x + (k - 1) * width, vals, width,
               label=cfg, color=CFG_COLORS[cfg], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs_p],
                       rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(f"{ylabel} score", fontsize=10)
    ax.set_title(f"QA — {ylabel} (XQuAD)", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    f"QA performance — F1 and Exact Match (layer {ABLATION_LAYER})",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_qa_f1_em.png"), dpi=150)
plt.close()
print("→ 4_qa_f1_em.png")


# ── Plot 5 — Summary: mean Δ per task × config ───────────────────────────────
summary_tasks = [
    ("QA (F1)",      results['qa'],             QA_LANGS),
    ("QA (EM)",      results['qa_em'],           QA_LANGS),
    ("Class. (Acc)", results['classification'],  CLASS_LANGS),
    ("Transl (BLEU)",results['translation'],     TRANSL_LANGS),
]
configs = ["top-1", "top-1+2"]
x       = np.arange(len(summary_tasks))
width   = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
for k, cfg in enumerate(configs):
    means = []
    stds  = []
    for _, task_res, langs in summary_tasks:
        deltas = [
            task_res[l].get(cfg, np.nan) - task_res[l].get("baseline", np.nan)
            for l in langs if l in task_res
        ]
        deltas = [d for d in deltas if not np.isnan(d)]
        means.append(np.mean(deltas) if deltas else 0)
        stds.append(np.std(deltas)   if deltas else 0)

    offset = (k - 0.5) * width
    ax.bar(x + offset, means, width, yerr=stds, capsize=4,
           label=cfg, color=CFG_COLORS[cfg], alpha=0.85, edgecolor='white')

ax.axhline(0, color='black', linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels([t for t, _, _ in summary_tasks], fontsize=10)
ax.set_ylabel("Mean Δ metric (ablated − baseline)", fontsize=10)
ax.set_title(
    f"Mean performance drop per task — layer {ABLATION_LAYER} ablation\n"
    "(error bars = std across languages)",
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_summary_mean_delta.png"), dpi=150)
plt.close()
print("→ 5_summary_mean_delta.png")


# ── CSV export ────────────────────────────────────────────────────────────────
rows = []
for task_key, task_res, langs, metric_name in [
    ("qa",             results['qa'],             QA_LANGS,     "F1"),
    ("qa_em",          results['qa_em'],           QA_LANGS,     "EM"),
    ("classification", results['classification'],  CLASS_LANGS,  "Accuracy"),
    ("translation",    results['translation'],     TRANSL_LANGS, "BLEU"),
]:
    for lang in langs:
        if lang not in task_res:
            continue
        base = task_res[lang].get("baseline", np.nan)
        for cfg in ABLATION_CONFIGS:
            val = task_res[lang].get(cfg, np.nan)
            rows.append({
                "task"   : task_key,
                "metric" : metric_name,
                "lang"   : lang,
                "config" : cfg,
                "value"  : round(val, 3),
                "delta"  : round(val - base, 3),
            })

df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(OUTPUT_DIR, "downstream_results.csv"), index=False)
print("→ downstream_results.csv")

print(f"\n✅ All outputs saved to '{OUTPUT_DIR}/'  —  5 plots + 1 CSV")

# Print summary table
print("\n── Summary table ──")
pivot = df_out[df_out.config != "baseline"].pivot_table(
    index=["task", "metric", "lang"], columns="config", values="delta"
)
print(pivot.to_string())