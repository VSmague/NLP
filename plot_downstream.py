"""
plot_downstream_results.py
===========================
Reads the downstream_results_layer10.pkl cache and generates
all plots — no model or GPU needed.

Run after downstream_ablation.py has computed at least QA + Classification.
Translation plots are skipped if not in cache.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR    = "sae_features"
OUTPUT_DIR  = "plots_downstream"
ABLATION_LAYER = 10
CACHE_PATH  = os.path.join(SAVE_DIR, f"downstream_results_layer{ABLATION_LAYER}.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese',
    'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic',
}

CFG_COLORS  = {"baseline": "#888888", "top-1+2": "#C44E52"}
CFG_LABELS  = {"baseline": "Baseline (no ablation)", "top-1+2": "Ablate top-1 & top-2"}
CFG_HATCHES = {"baseline": "", "top-1+2": "//"}

# ── Load cache ────────────────────────────────────────────────────────────────
print(f"Loading {CACHE_PATH}...")
with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)

results        = cache["results"]
ABLATION_LAYER = cache.get("ablation_layer", ABLATION_LAYER)
N_SAMPLES      = cache.get("n_samples", "?")

# Detect available tasks
HAS_QA     = bool(results.get("qa"))
HAS_CLS    = bool(results.get("classification"))
HAS_TRANSL = bool(results.get("translation"))

print(f"  Layer: {ABLATION_LAYER}  |  N_samples: {N_SAMPLES}")
print(f"  QA: {HAS_QA}  Classification: {HAS_CLS}  Translation: {HAS_TRANSL}")

QA_LANGS    = sorted(results["qa"].keys())        if HAS_QA    else []
CLASS_LANGS = sorted(results["classification"].keys()) if HAS_CLS else []
TRANSL_LANGS= sorted(results["translation"].keys())    if HAS_TRANSL else []

CONFIGS = ["baseline", "top-1+2"]


# =============================================================================
# Helpers
# =============================================================================

def get_val(task_res, lang, cfg):
    return task_res.get(lang, {}).get(cfg, np.nan)

def get_delta(task_res, lang, cfg="top-1+2"):
    base = get_val(task_res, lang, "baseline")
    abl  = get_val(task_res, lang, cfg)
    return abl - base

def bar_group(ax, langs, task_res, ylabel, title, pct=False):
    """Grouped bar chart: baseline vs ablated, one group per language."""
    x     = np.arange(len(langs))
    width = 0.38
    for k, cfg in enumerate(CONFIGS):
        vals   = [get_val(task_res, l, cfg) for l in langs]
        offset = (k - 0.5) * width
        ax.bar(x + offset, vals, width,
               label=CFG_LABELS[cfg],
               color=CFG_COLORS[cfg],
               hatch=CFG_HATCHES[cfg],
               alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs],
                       rotation=30, ha='right', fontsize=9)
    suffix = " (%)" if pct else ""
    ax.set_ylabel(f"{ylabel}{suffix}", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def delta_bars(ax, langs, task_res, ylabel, title):
    """Single bar per language showing Δ = ablated − baseline."""
    x      = np.arange(len(langs))
    deltas = [get_delta(task_res, l) for l in langs]
    colors = ['#C44E52' if d < 0 else '#55A868' for d in deltas]
    bars   = ax.bar(x, deltas, color=colors, alpha=0.85, edgecolor='white', width=0.6)

    # Value labels on bars
    for bar, d in zip(bars, deltas):
        if not np.isnan(d):
            va  = 'bottom' if d >= 0 else 'top'
            off = 0.3 if d >= 0 else -0.3
            ax.text(bar.get_x() + bar.get_width()/2,
                    d + off * 0.05,
                    f"{d:+.1f}", ha='center', va=va, fontsize=8, fontweight='bold')

    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f"Δ {ylabel}  (ablated − baseline)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    red_patch   = mpatches.Patch(color='#C44E52', label='Degradation')
    green_patch = mpatches.Patch(color='#55A868', label='Improvement')
    ax.legend(handles=[red_patch, green_patch], fontsize=8)


# =============================================================================
# PLOT 1 — Absolute metrics: QA F1 + Classification Acc
# =============================================================================

n_plots = sum([HAS_QA, HAS_CLS, HAS_TRANSL])
if n_plots == 0:
    print("No results to plot!")
    exit()

tasks_to_plot = []
if HAS_QA:
    tasks_to_plot.append(("QA — F1 (XQuAD)", results["qa"], QA_LANGS, "F1 score"))
if HAS_CLS:
    tasks_to_plot.append(("Classification — Accuracy (Amazon Reviews)", results["classification"], CLASS_LANGS, "Accuracy (%)"))
if HAS_TRANSL:
    tasks_to_plot.append(("Translation — BLEU (→ EN)", results["translation"], TRANSL_LANGS, "BLEU"))

fig, axes = plt.subplots(1, len(tasks_to_plot), figsize=(7 * len(tasks_to_plot), 6))
if len(tasks_to_plot) == 1:
    axes = [axes]

for ax, (title, task_res, langs, ylabel) in zip(axes, tasks_to_plot):
    bar_group(ax, langs, task_res, ylabel, title, pct=("%" in ylabel))

fig.suptitle(
    f"Downstream performance — baseline vs ablation of top-1 & top-2 features (layer {ABLATION_LAYER})\n"
    f"Model: Qwen3-0.6B  |  {N_SAMPLES} samples/language",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_absolute_metrics.png"), dpi=150, bbox_inches='tight')
plt.close()
print("→ 1_absolute_metrics.png")


# =============================================================================
# PLOT 2 — Δ metrics per language (red = degradation)
# =============================================================================

fig, axes = plt.subplots(1, len(tasks_to_plot), figsize=(7 * len(tasks_to_plot), 6))
if len(tasks_to_plot) == 1:
    axes = [axes]

delta_titles = [
    ("ΔF1 — QA (XQuAD)",           results["qa"],             QA_LANGS,     "F1"),
    ("ΔAcc — Classification",       results["classification"], CLASS_LANGS,  "Accuracy (%)"),
    ("ΔBLEU — Translation (→ EN)",  results["translation"],    TRANSL_LANGS, "BLEU"),
]
delta_titles = [(t, r, l, m) for (t, r, l, m) in delta_titles
                if (t.startswith("ΔF1") and HAS_QA) or
                   (t.startswith("ΔAcc") and HAS_CLS) or
                   (t.startswith("ΔBLEU") and HAS_TRANSL)]

for ax, (title, task_res, langs, metric) in zip(axes, delta_titles):
    delta_bars(ax, langs, task_res, metric, title)

fig.suptitle(
    f"Performance drop after ablating top-1 & top-2 features (layer {ABLATION_LAYER})\n"
    "Negative = ablation hurts  |  Positive = unexpected improvement",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_delta_per_language.png"), dpi=150, bbox_inches='tight')
plt.close()
print("→ 2_delta_per_language.png")


# =============================================================================
# PLOT 3 — QA: F1 and EM side by side (if available)
# =============================================================================

if HAS_QA and results.get("qa_em"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_group(axes[0], QA_LANGS, results["qa"],    "F1 score",   "QA — F1")
    bar_group(axes[1], QA_LANGS, results["qa_em"], "Exact Match","QA — Exact Match")
    fig.suptitle(f"QA metrics — F1 and Exact Match (layer {ABLATION_LAYER})",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_qa_f1_em.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("→ 3_qa_f1_em.png")


# =============================================================================
# PLOT 4 — Heatmap: Δ across all tasks × languages
# =============================================================================

# Build unified table
all_rows = []
if HAS_QA:
    for l in QA_LANGS:
        all_rows.append({"task": "QA (F1)", "lang": LANG_NAMES[l],
                         "delta": get_delta(results["qa"], l)})
if HAS_QA and results.get("qa_em"):
    for l in QA_LANGS:
        all_rows.append({"task": "QA (EM)", "lang": LANG_NAMES[l],
                         "delta": get_delta(results["qa_em"], l)})
if HAS_CLS:
    for l in CLASS_LANGS:
        all_rows.append({"task": "Classification", "lang": LANG_NAMES[l],
                         "delta": get_delta(results["classification"], l)})
if HAS_TRANSL:
    for l in TRANSL_LANGS:
        all_rows.append({"task": "Translation (BLEU)", "lang": LANG_NAMES[l],
                         "delta": get_delta(results["translation"], l)})

df = pd.DataFrame(all_rows)
pivot = df.pivot(index="task", columns="lang", values="delta")

fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), 4))
vmax = np.nanmax(np.abs(pivot.values))
im   = ax.imshow(pivot.values, cmap='RdBu', aspect='auto',
                 vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontsize=9)
ax.set_yticklabels(pivot.index, fontsize=10)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                    fontsize=8.5,
                    color='white' if abs(val) > vmax * 0.55 else 'black',
                    fontweight='bold')

plt.colorbar(im, ax=ax, label="Δ metric (ablated − baseline)")
ax.set_title(
    f"Δ performance heatmap — ablate top-1+2 features (layer {ABLATION_LAYER})\n"
    "Blue = degradation  |  Red = improvement",
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_delta_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("→ 4_delta_heatmap.png")


# =============================================================================
# PLOT 5 — Summary bar: mean Δ per task
# =============================================================================

summary = []
if HAS_QA:
    d = [get_delta(results["qa"], l) for l in QA_LANGS]
    summary.append(("QA (F1)",         np.nanmean(d), np.nanstd(d)))
if HAS_QA and results.get("qa_em"):
    d = [get_delta(results["qa_em"], l) for l in QA_LANGS]
    summary.append(("QA (EM)",         np.nanmean(d), np.nanstd(d)))
if HAS_CLS:
    d = [get_delta(results["classification"], l) for l in CLASS_LANGS]
    summary.append(("Classification",  np.nanmean(d), np.nanstd(d)))
if HAS_TRANSL:
    d = [get_delta(results["translation"], l) for l in TRANSL_LANGS]
    summary.append(("Translation",     np.nanmean(d), np.nanstd(d)))

fig, ax = plt.subplots(figsize=(8, 5))
labels = [s[0] for s in summary]
means  = [s[1] for s in summary]
stds   = [s[2] for s in summary]
colors = ['#C44E52' if m < 0 else '#55A868' for m in means]

bars = ax.bar(range(len(summary)), means, yerr=stds, capsize=6,
              color=colors, alpha=0.85, edgecolor='white', width=0.5)

for bar, m in zip(bars, means):
    va  = 'bottom' if m >= 0 else 'top'
    off = max(stds) * 0.15
    ax.text(bar.get_x() + bar.get_width()/2,
            m + (off if m >= 0 else -off),
            f"{m:+.2f}", ha='center', va=va, fontsize=10, fontweight='bold')

ax.axhline(0, color='black', linewidth=1.0)
ax.set_xticks(range(len(summary)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Mean Δ metric  (ablated − baseline)", fontsize=10)
ax.set_title(
    f"Mean performance drop across languages — layer {ABLATION_LAYER} ablation\n"
    "(error bars = std across languages)",
    fontsize=12, fontweight='bold'
)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_summary_mean_delta.png"), dpi=150, bbox_inches='tight')
plt.close()
print("→ 5_summary_mean_delta.png")


# =============================================================================
# CSV export + console summary
# =============================================================================

rows = []
task_map = []
if HAS_QA:
    task_map += [("qa", "F1", results["qa"], QA_LANGS),
                 ("qa_em", "EM", results.get("qa_em", {}), QA_LANGS)]
if HAS_CLS:
    task_map.append(("classification", "Accuracy", results["classification"], CLASS_LANGS))
if HAS_TRANSL:
    task_map.append(("translation", "BLEU", results["translation"], TRANSL_LANGS))

for task_key, metric, task_res, langs in task_map:
    for lang in langs:
        base = get_val(task_res, lang, "baseline")
        abl  = get_val(task_res, lang, "top-1+2")
        rows.append({
            "task": task_key, "metric": metric, "language": LANG_NAMES.get(lang, lang),
            "baseline": round(base, 3), "ablated": round(abl, 3),
            "delta": round(abl - base, 3),
            "delta_pct": round((abl - base) / base * 100, 1) if base and base != 0 else np.nan,
        })

df_out = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "downstream_results.csv")
df_out.to_csv(csv_path, index=False)
print(f"→ downstream_results.csv")

print("\n── Results summary ──")
print(df_out.to_string(index=False))

print(f"\n✅ All plots saved to '{OUTPUT_DIR}/'")