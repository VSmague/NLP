"""
reproduce_figures_section5.py
==============================
Reproduces Figures 5 and 6 from Section 5 of:
  "Unveiling Language-Specific Features in LLMs via Sparse Autoencoders"
  Deng et al., ACL 2025

Requires: having run compute_nu_scores.py once (sae_features/ must exist).
"""

import os
import json
import pickle
import configparser
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# ── Auth ──────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("secrets.ini")
os.environ["HF_TOKEN"] = config["huggingface"]["token"]

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-0.6B"
SAE_RELEASE = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR    = "sae_features"
OUTPUT_DIR  = "output/figures_section5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FIG5 = os.path.join(SAVE_DIR, "fig5_cache.pkl")
CACHE_FIG6 = os.path.join(SAVE_DIR, "fig6_cache.pkl")

N_SENTENCES = 100
RECOMPUTE   = False

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish',    'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean',    'pt': 'Portuguese',
    'th': 'Thai',     'vi': 'Vietnamese','zh': 'Chinese', 'ar': 'Arabic',
}

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Time utilities ────────────────────────────────────────────────────────────

def fmt_time(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

def progress(done, total, t0, prefix=""):
    elapsed = time.time() - t0
    eta     = (elapsed / done) * (total - done) if done > 0 else 0
    pct     = done / total * 100
    bar     = "█" * int(30 * done / total) + "░" * (30 - int(30 * done / total))
    print(f"\r{prefix} [{bar}] {pct:5.1f}%  elapsed={fmt_time(elapsed)}  ETA={fmt_time(eta)}",
          end="", flush=True)
    if done == total:
        print()


# =============================================================================
# Load SAE cache
# =============================================================================

print("\n[0] Loading SAE cache...")
with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)

LAYERS = meta["layers"]

top_index_all = {}
for layer in LAYERS:
    top_index_all[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"), weights_only=True
    )
print(f"   {len(LAYERS)} layers loaded: {LAYERS}")


# =============================================================================
# Core utilities
# =============================================================================

def directional_ablation_forward(model, sae, layer, feature_indices, inputs):
    W_dec     = sae.W_dec.T.to(device)
    dirs      = W_dec[:, feature_indices]
    norms     = torch.norm(dirs, dim=0) ** 2
    dirs_norm = dirs / norms.clamp(min=1e-8)

    def _hook(module, inp, output):
        if isinstance(output, tuple):
            act = output[0].to(torch.float32)
        else:
            act = output.to(torch.float32)
        coeff       = act @ dirs_norm
        act_ablated = act - coeff @ dirs_norm.T
        if isinstance(output, tuple):
            return (act_ablated.to(output[0].dtype),) + output[1:]
        return act_ablated.to(output.dtype)

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        out = model.forward(inputs)
    finally:
        handle.remove()
    return out


def compute_ce_loss(logits, inputs):
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = inputs[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels)


def compute_delta_ce(model, sae, layer, feat_idx, sentences, tokenizer):
    deltas = []
    for text in sentences:
        inp = tokenizer.encode(text, return_tensors='pt',
                               add_special_tokens=True).to(device)
        if inp.size(1) < 2:
            continue
        base = compute_ce_loss(model.forward(inp)["logits"], inp).item()
        abl  = compute_ce_loss(
            directional_ablation_forward(model, sae, layer, feat_idx, inp)["logits"], inp
        ).item()
        deltas.append(abl - base)
    return float(np.nanmean(deltas)) if deltas else 0.0


# =============================================================================
# Load model + data (only if needed)
# =============================================================================

need_compute = (RECOMPUTE or
                not os.path.exists(CACHE_FIG5) or
                not os.path.exists(CACHE_FIG6))

if need_compute:
    import requests

    print("\n[1] Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    print(f"   Loaded in {fmt_time(time.time()-t0)}")

    print("\n[2] Loading data...")
    DATA_URL = ("https://raw.githubusercontent.com/Aatrox103/"
                "multilingual-llm-features/main/SAE/data/multilingual_data.jsonl")
    resp = requests.get(DATA_URL)
    resp.raise_for_status()
    multilingual_texts = []
    for line in resp.text.strip().split('\n'):
        obj = json.loads(line)
        multilingual_texts.append(obj['text'] if isinstance(obj, dict) else obj)

    sentences_per_lang = {
        lang: multilingual_texts[i * 100 : i * 100 + N_SENTENCES]
        for i, lang in enumerate(TARGET_LANGUAGES)
    }
    print(f"   {len(multilingual_texts)} sentences loaded")


# =============================================================================
# FIGURE 5 — compute
# -----------------------------------------------------------------------------
# For each language L, for each config in FIG5_CONFIGS:
#   Ablate top-1 OR top-1+2 features of L
#   Measure delta-CE on ALL languages, at EACH layer separately
#
# fig5_data[config][lang_ablate][lang_eval] = list of delta-CE per layer
#
# FIG5_CONFIGS — change here to switch between top-1 and top-1+2:
#   "top-1"   : ablate only the rank-1 feature
#   "top-1+2" : ablate rank-1 and rank-2 together
# =============================================================================

FIG5_CONFIGS = ["top-1", "top-1+2"]

if RECOMPUTE or not os.path.exists(CACHE_FIG5):
    print("\n[Fig 5] Computing delta-CE per language, per layer, per config...")

    fig5_data = {
        cfg: {la: {le: [] for le in TARGET_LANGUAGES} for la in TARGET_LANGUAGES}
        for cfg in FIG5_CONFIGS
    }

    t0    = time.time()
    total = len(TARGET_LANGUAGES) * len(LAYERS)
    done  = 0

    for lang_ablate in TARGET_LANGUAGES:
        j = TARGET_LANGUAGES.index(lang_ablate)

        for layer in LAYERS:
            sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)

            feat_map = {
                "top-1"   : top_index_all[layer][j, 0:1].to(device),
                "top-1+2" : top_index_all[layer][j, 0:2].to(device),
            }

            for lang_eval in TARGET_LANGUAGES:
                sents = sentences_per_lang[lang_eval]
                for cfg in FIG5_CONFIGS:
                    delta = compute_delta_ce(model, sae, layer, feat_map[cfg], sents, tokenizer)
                    fig5_data[cfg][lang_ablate][lang_eval].append(delta)

            del sae
            done += 1
            progress(done, total, t0, prefix="[Fig5]")

    with open(CACHE_FIG5, "wb") as f:
        pickle.dump({"fig5_data": fig5_data, "layers": LAYERS, "configs": FIG5_CONFIGS}, f)
    print(f"\n   Done — {fmt_time(time.time()-t0)}")

else:
    print("\n[Fig 5] Loading from cache...")
    with open(CACHE_FIG5, "rb") as f:
        cache5 = pickle.load(f)
    fig5_data    = cache5["fig5_data"]
    LAYERS       = cache5.get("layers", LAYERS)
    FIG5_CONFIGS = cache5.get("configs", ["top-1", "top-1+2"])
    print("   Loaded.")


# =============================================================================
# FIGURE 6 — compute
# -----------------------------------------------------------------------------
# For EVERY language as target, compute delta-CE for:
#   top-1 only | top-2 only | top-1+2 together
# Evaluated on: the target language itself + one typologically close language
#               + one typologically distant language
#
# Typological proximity pairs used:
#   en  → close: fr (Indo-European),       distant: zh (Sino-Tibetan)
#   es  → close: pt (Ibero-Romance),       distant: ja (Japonic)
#   fr  → close: es (Romance),             distant: th (Tai-Kadai)
#   ja  → close: ko (similar morphology),  distant: en (Germanic)
#   ko  → close: ja (agglutinative),       distant: ar (Semitic)
#   pt  → close: es (Ibero-Romance),       distant: th (Tai-Kadai)
#   th  → close: vi (mainland SE Asia),    distant: ar (Semitic)
#   vi  → close: th (mainland SE Asia),    distant: en (Germanic)
#   zh  → close: ja (shared script),       distant: es (Romance)
#   ar  → close: en (shared loanwords),    distant: zh (Sino-Tibetan)
# =============================================================================

LANG_CLOSE = {
    'en': 'fr',  'es': 'pt',  'fr': 'es',  'ja': 'ko',  'ko': 'ja',
    'pt': 'es',  'th': 'vi',  'vi': 'th',  'zh': 'ja',  'ar': 'en',
}
LANG_DISTANT = {
    'en': 'zh',  'es': 'ja',  'fr': 'th',  'ja': 'en',  'ko': 'ar',
    'pt': 'th',  'th': 'ar',  'vi': 'en',  'zh': 'es',  'ar': 'zh',
}

# All unique (target, close, distant) triples
FIG6_TRIPLES = {
    lang: (lang, LANG_CLOSE[lang], LANG_DISTANT[lang])
    for lang in TARGET_LANGUAGES
}
# All languages we need to evaluate across all triples
ALL_EVAL_LANGS = sorted(set(
    l for triple in FIG6_TRIPLES.values() for l in triple
))

if RECOMPUTE or not os.path.exists(CACHE_FIG6):
    print(f"\n[Fig 6] Computing synergy delta-CE for ALL target languages...")
    print(f"   Configs: top-1 | top-2 | top-1+2")
    print(f"   {len(TARGET_LANGUAGES)} targets × {len(LAYERS)} layers × 3 eval langs × 3 configs\n")

    # fig6_data[lang_ablate][config][lang_eval] = [delta per layer]
    fig6_data = {
        la: {
            "top-1"   : {le: [] for le in ALL_EVAL_LANGS},
            "top-2"   : {le: [] for le in ALL_EVAL_LANGS},
            "top-1+2" : {le: [] for le in ALL_EVAL_LANGS},
        }
        for la in TARGET_LANGUAGES
    }

    t0    = time.time()
    total = len(TARGET_LANGUAGES) * len(LAYERS)
    done  = 0

    for lang_ablate in TARGET_LANGUAGES:
        j     = TARGET_LANGUAGES.index(lang_ablate)
        triple = FIG6_TRIPLES[lang_ablate]   # (target, close, distant)

        for layer in LAYERS:
            sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)

            feat_top1 = top_index_all[layer][j, 0:1].to(device)
            feat_top2 = top_index_all[layer][j, 1:2].to(device)
            feat_both = top_index_all[layer][j, 0:2].to(device)

            for lang_eval in triple:
                sents = sentences_per_lang[lang_eval]
                fig6_data[lang_ablate]["top-1"  ][lang_eval].append(
                    compute_delta_ce(model, sae, layer, feat_top1, sents, tokenizer))
                fig6_data[lang_ablate]["top-2"  ][lang_eval].append(
                    compute_delta_ce(model, sae, layer, feat_top2, sents, tokenizer))
                fig6_data[lang_ablate]["top-1+2"][lang_eval].append(
                    compute_delta_ce(model, sae, layer, feat_both, sents, tokenizer))

            del sae
            done += 1
            progress(done, total, t0, prefix="[Fig6]")

    with open(CACHE_FIG6, "wb") as f:
        pickle.dump({
            "fig6_data"   : fig6_data,
            "layers"      : LAYERS,
            "fig6_triples": FIG6_TRIPLES,
        }, f)
    print(f"\n   Done — {fmt_time(time.time()-t0)}")

else:
    print("\n[Fig 6] Loading from cache...")
    with open(CACHE_FIG6, "rb") as f:
        cache6 = pickle.load(f)
    fig6_data    = cache6["fig6_data"]
    LAYERS       = cache6["layers"]
    FIG6_TRIPLES = cache6.get("fig6_triples", FIG6_TRIPLES)
    print("   Loaded.")


# =============================================================================
# PLOT — Figure 5  (per-layer line plot, one subplot per ablated language)
# -----------------------------------------------------------------------------
# Each subplot = one ablated language
# X axis = layer index
# One line per evaluated language
# Target language line is thick red, others are thin grey
# =============================================================================

print("\n[Plot] Generating Figure 5 (per-layer, one plot per config)...")

LANGS_PLOT = [l for l in TARGET_LANGUAGES if l != 'en']

COLOR_TARGET = '#D62728'
COLOR_EN     = '#888888'
COLOR_OTHER  = '#AEC6CF'

# Config display names for titles and filenames
CFG_LABEL = {
    "top-1"   : "top-1 feature",
    "top-1+2" : "top-1 & top-2 features",
}
CFG_FNAME = {
    "top-1"   : "figure5_ablation_top1_per_layer.png",
    "top-1+2" : "figure5_ablation_top1_top2_per_layer.png",
}

for cfg in FIG5_CONFIGS:
    n_cols = 3
    n_rows = int(np.ceil(len(LANGS_PLOT) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                              gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
    axes = axes.flatten()

    for ax_idx, lang_ablate in enumerate(LANGS_PLOT):
        ax = axes[ax_idx]

        for lang_eval in TARGET_LANGUAGES:
            y = fig5_data[cfg][lang_ablate][lang_eval]

            if lang_eval == lang_ablate:
                ax.plot(LAYERS, y, color=COLOR_TARGET, linewidth=2.5,
                        marker='o', markersize=4, zorder=5,
                        label=f"{LANG_NAMES[lang_eval]} (target)")
            elif lang_eval == 'en':
                ax.plot(LAYERS, y, color=COLOR_EN, linewidth=1.0,
                        linestyle='--', marker='x', markersize=3, zorder=2,
                        label="English")
            else:
                ax.plot(LAYERS, y, color=COLOR_OTHER, linewidth=0.9,
                        alpha=0.7, marker='.', markersize=2, zorder=1)

        ax.axhline(0, color='black', linestyle=':', linewidth=0.7)
        ax.set_title(f"Ablate {LANG_NAMES[lang_ablate]} {CFG_LABEL[cfg]}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("ΔCE loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles = [
            plt.Line2D([0], [0], color=COLOR_TARGET, linewidth=2.0,
                       label=f"{LANG_NAMES[lang_ablate]} (target)"),
            plt.Line2D([0], [0], color=COLOR_OTHER,  linewidth=1.0,
                       label="Other languages"),
            plt.Line2D([0], [0], color=COLOR_EN,     linewidth=1.0,
                       linestyle='--', label="English"),
        ]
        ax.legend(handles=handles, fontsize=6.5, loc='upper left', framealpha=0.6)

    for ax_idx in range(len(LANGS_PLOT), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        f"Figure 5 — ΔCE per layer after ablating {CFG_LABEL[cfg]}\n"
        "Ablation significantly impacts the target language (red) but not others",
        fontsize=11, fontweight='bold', y=1.01
    )

    fname = CFG_FNAME[cfg]
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> {fname}")


# =============================================================================
# PLOT — Figure 6
# -----------------------------------------------------------------------------
# One separate figure per target language.
# Each figure has 3 subplots side by side:
#   col 0 : target language itself        (e.g. French)
#   col 1 : typologically close language  (e.g. Spanish)
#   col 2 : typologically distant language(e.g. Arabic)
#
# Each subplot shows 3 curves (top-1, top-2, top-1+2) across layers.
# Saved as: figure6_synergy_<target_lang>.png
# =============================================================================

print("[Plot] Generating Figure 6 (one plot per target language)...")

CONFIG_COLORS = {
    "top-1"   : "#4C72B0",
    "top-2"   : "#DD8452",
    "top-1+2" : "#C44E52",
}
CONFIG_LABELS = {
    "top-1"   : "Ablate top-1 only",
    "top-2"   : "Ablate top-2 only",
    "top-1+2" : "Ablate top-1 & top-2",
}
CONFIG_STYLES = {"top-1": "-", "top-2": "--", "top-1+2": "-"}
CONFIG_WIDTHS = {"top-1": 1.8, "top-2": 1.8, "top-1+2": 2.5}

# Role labels shown in subplot titles
ROLE_LABEL = {0: "Target", 1: "Typologically close", 2: "Typologically distant"}
ROLE_TITLE_COLOR = {0: '#D62728', 1: '#2CA02C', 2: '#7B4173'}

for lang_ablate in TARGET_LANGUAGES:
    lang_target, lang_close, lang_distant = FIG6_TRIPLES[lang_ablate]
    eval_langs = [lang_target, lang_close, lang_distant]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for col, lang_eval in enumerate(eval_langs):
        ax = axes[col]

        for config in ["top-1", "top-2", "top-1+2"]:
            vals = fig6_data[lang_ablate][config][lang_eval]
            ax.plot(LAYERS, vals,
                    color=CONFIG_COLORS[config],
                    linestyle=CONFIG_STYLES[config],
                    linewidth=CONFIG_WIDTHS[config],
                    marker='o', markersize=4,
                    label=CONFIG_LABELS[config])

        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("ΔCE loss", fontsize=10)
        ax.grid(linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

        role_str  = ROLE_LABEL[col]
        ax.set_title(
            f"{role_str}\nEval on {LANG_NAMES[lang_eval]}",
            fontsize=10, fontweight='bold',
            color=ROLE_TITLE_COLOR[col]
        )

        # Annotate synergy on the target subplot
        if col == 0:
            best_idx = int(np.argmax(fig6_data[lang_ablate]["top-1+2"][lang_eval]))
            best_val = fig6_data[lang_ablate]["top-1+2"][lang_eval][best_idx]
            top1_val = fig6_data[lang_ablate]["top-1"][lang_eval][best_idx]
            top2_val = fig6_data[lang_ablate]["top-2"][lang_eval][best_idx]
            synergy  = best_val - (top1_val + top2_val)
            if synergy > 0.01:
                ax.annotate(
                    f"synergy\n+{synergy:.3f}",
                    xy=(LAYERS[best_idx], best_val),
                    xytext=(LAYERS[best_idx] + 1, best_val + 0.05),
                    fontsize=7.5, color=CONFIG_COLORS["top-1+2"],
                    arrowprops=dict(arrowstyle='->', color=CONFIG_COLORS["top-1+2"], lw=1.2),
                )

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.5, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        f"Figure 6 — Ablating {LANG_NAMES[lang_ablate]} top-1 & top-2 features\n"
        f"Synergy expected on {LANG_NAMES[lang_target]}, "
        f"not on close ({LANG_NAMES[lang_close]}) "
        f"or distant ({LANG_NAMES[lang_distant]})",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    fname = f"figure6_synergy_{lang_ablate}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   -> {fname}")

print(f"\nAll figures saved to '{OUTPUT_DIR}/'")
print("   figure5_ablation_per_layer.png")
for lang in TARGET_LANGUAGES:
    print(f"   figure6_synergy_{lang}.png")