"""
cluster_synergy_from_cache_gemma.py
=====================================
Analyse la synergie intra-cluster vs inter-cluster pour Gemma-2-2B,
EN RÉUTILISANT le cache sae_features_gemma/fig6_cache.pkl déjà calculé.

Clusters obtenus sur Gemma (à mettre à jour si différents) :
  1: ['th']
  2: ['ar']
  3: ['en', 'es', 'fr', 'pt', 'vi']
  4: ['ja', 'zh']
  5: ['ko']

Paires de proximité typologique utilisées lors du calcul (Gemma) :
  en  → close: fr,  distant: zh
  es  → close: pt,  distant: ja
  fr  → close: es,  distant: ar   ← note: th dans Qwen, ar ici
  ja  → close: ko,  distant: en
  ko  → close: ja,  distant: ar
  pt  → close: es,  distant: th
  th  → close: vi,  distant: ar
  vi  → close: th,  distant: en
  zh  → close: ja,  distant: es
  ar  → close: en,  distant: zh
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Parameters ────────────────────────────────────────────────────────────────
SAVE_DIR   = "sae_features_gemma"
OUTPUT_DIR = "plots_synergy_gemma"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FIG6 = os.path.join(SAVE_DIR, "fig6_cache.pkl")

# ── Clusters ──────────────────────────────────────────────────────────────────
# Update these if your Gemma clustering gives different results
CLUSTERS = {
    1: ['th'],
    2: ['ar'],
    3: ['en', 'es', 'fr', 'pt', 'vi'],
    4: ['ja', 'zh'],
    5: ['ko'],
}
LANG_TO_CLUSTER = {l: cid for cid, langs in CLUSTERS.items() for l in langs}
CLUSTER_COLORS  = {
    1: '#64B5CD',
    2: '#C44E52',
    3: '#55A868',
    4: '#CCB974',
    5: '#8172B3',
}

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese',
    'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic',
}

# Paires de proximité — identiques à reproduce_figures_section5.py (Gemma version)
LANG_CLOSE = {
    'en': 'fr',  'es': 'pt',  'fr': 'es',  'ja': 'ko',  'ko': 'ja',
    'pt': 'es',  'th': 'vi',  'vi': 'th',  'zh': 'ja',  'ar': 'en',
}
LANG_DISTANT = {
    'en': 'zh',  'es': 'ja',  'fr': 'ar',  'ja': 'en',  'ko': 'ar',
    'pt': 'th',  'th': 'ar',  'vi': 'en',  'zh': 'es',  'ar': 'zh',
}

FIG6_TRIPLES = {
    lang: (lang, LANG_CLOSE[lang], LANG_DISTANT[lang])
    for lang in TARGET_LANGUAGES
}

MODEL_LABEL = "Gemma-2-2B"


# =============================================================================
# Load cache
# =============================================================================

print(f"Loading {CACHE_FIG6}...")
with open(CACHE_FIG6, "rb") as f:
    cache = pickle.load(f)

fig6_data    = cache["fig6_data"]
LAYERS       = cache["layers"]
FIG6_TRIPLES = cache.get("fig6_triples", FIG6_TRIPLES)
n_layers     = len(LAYERS)
print(f"  {len(fig6_data)} languages × {n_layers} layers loaded")
print(f"  Layers: {LAYERS}")


# =============================================================================
# Compute synergy arrays
# =============================================================================

records = []

for lang_ablate in TARGET_LANGUAGES:
    lang_target, lang_close, lang_distant = FIG6_TRIPLES[lang_ablate]

    for role, lang_eval in [("target",  lang_target),
                             ("close",   lang_close),
                             ("distant", lang_distant)]:

        d1  = np.array(fig6_data[lang_ablate]["top-1"  ][lang_eval])
        d2  = np.array(fig6_data[lang_ablate]["top-2"  ][lang_eval])
        d12 = np.array(fig6_data[lang_ablate]["top-1+2"][lang_eval])
        syn = d12 - (d1 + d2)

        cluster_rel = (
            "intra" if LANG_TO_CLUSTER[lang_ablate] == LANG_TO_CLUSTER[lang_eval]
            else "inter"
        )

        records.append({
            "lang_ablate"  : lang_ablate,
            "lang_eval"    : lang_eval,
            "role"         : role,
            "cluster_rel"  : cluster_rel,
            "delta_top1"   : d1,
            "delta_top2"   : d2,
            "delta_top1p2" : d12,
            "synergy"      : syn,
            "syn_mean"     : float(syn.mean()),
        })

print(f"  {len(records)} (lang_ablate, lang_eval) records built\n")


# =============================================================================
# PLOT 1 — Synergy per layer: target vs close vs distant
# =============================================================================

def avg_synergy_by_role(role):
    sel = [r for r in records if r["role"] == role]
    mat = np.stack([r["synergy"] for r in sel])
    return mat.mean(axis=0), mat.std(axis=0)

role_styles = {
    "target" : dict(color='#C44E52', label='Target language (same as ablated)', lw=2.4, ls='-',  marker='o'),
    "close"  : dict(color='#55A868', label='Typologically close language',       lw=1.8, ls='--', marker='s'),
    "distant": dict(color='#4C72B0', label='Typologically distant language',     lw=1.8, ls=':',  marker='^'),
}

fig, ax = plt.subplots(figsize=(10, 5))
for role, sty in role_styles.items():
    mean, std = avg_synergy_by_role(role)
    ax.plot(LAYERS, mean, color=sty['color'], linewidth=sty['lw'],
            linestyle=sty['ls'], marker=sty['marker'], markersize=5,
            label=sty['label'])
    ax.fill_between(LAYERS, mean - std, mean + std,
                    color=sty['color'], alpha=0.12)

ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Mean synergy  ΔCE_(1+2) − (ΔCE_1 + ΔCE_2)", fontsize=10)
ax.set_title(f"[{MODEL_LABEL}] Feature synergy per layer — target vs close vs distant\n"
             "(avg over all 10 ablation languages)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_synergy_by_role_per_layer.png"), dpi=150)
plt.close()
print("→ 1_synergy_by_role_per_layer.png")


# =============================================================================
# PLOT 2 — Intra-cluster vs inter-cluster synergy per layer
# =============================================================================

def avg_synergy_by_cluster_rel(rel):
    sel = [r for r in records if r["cluster_rel"] == rel]
    mat = np.stack([r["synergy"] for r in sel])
    return mat.mean(axis=0), mat.std(axis=0)

intra_mean, intra_std = avg_synergy_by_cluster_rel("intra")
inter_mean, inter_std = avg_synergy_by_cluster_rel("inter")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(LAYERS, intra_mean, color='#C44E52', linewidth=2.2, marker='o',
        markersize=5, label='Intra-cluster pairs')
ax.fill_between(LAYERS, intra_mean - intra_std, intra_mean + intra_std,
                color='#C44E52', alpha=0.15)
ax.plot(LAYERS, inter_mean, color='#4C72B0', linewidth=2.2, marker='s',
        markersize=5, label='Inter-cluster pairs')
ax.fill_between(LAYERS, inter_mean - inter_std, inter_mean + inter_std,
                color='#4C72B0', alpha=0.15)

ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Mean synergy  ΔCE_(1+2) − (ΔCE_1 + ΔCE_2)", fontsize=10)
ax.set_title(f"[{MODEL_LABEL}] Intra-cluster vs inter-cluster synergy per layer\n"
             "(ablate top-1 & top-2 of the same language)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_synergy_intra_vs_inter_per_layer.png"), dpi=150)
plt.close()
print("→ 2_synergy_intra_vs_inter_per_layer.png")


# =============================================================================
# PLOT 3 — Per-language bar chart: mean synergy for target / close / distant
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(len(TARGET_LANGUAGES))
width = 0.28

for offset, role, color, label in [
    (-width, "target",  '#C44E52', 'Target'),
    (0,      "close",   '#55A868', 'Close'),
    (+width, "distant", '#4C72B0', 'Distant'),
]:
    means = []
    for lang in TARGET_LANGUAGES:
        rec = next((r for r in records
                    if r["lang_ablate"] == lang and r["role"] == role), None)
        means.append(rec["syn_mean"] if rec else 0.0)
    ax.bar(x + offset, means, width, label=label, color=color,
           alpha=0.85, edgecolor='white')

# Color x-tick labels by cluster
ax.set_xticks(x)
xlabels = ax.set_xticklabels(
    [LANG_NAMES[l] for l in TARGET_LANGUAGES],
    rotation=30, ha='right', fontsize=9
)
for lbl, lang in zip(xlabels, TARGET_LANGUAGES):
    lbl.set_color(CLUSTER_COLORS[LANG_TO_CLUSTER[lang]])

ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax.set_ylabel("Mean synergy (avg over layers)", fontsize=10)
ax.set_title(f"[{MODEL_LABEL}] Feature synergy per ablation language\n"
             "Target / Typologically close / Typologically distant",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Cluster legend
cluster_patches = [
    mpatches.Patch(color=CLUSTER_COLORS[c],
                   label=f"C{c}: {', '.join(l.upper() for l in langs)}")
    for c, langs in CLUSTERS.items()
]
ax.legend(
    handles=ax.get_legend_handles_labels()[0] + cluster_patches,
    labels=ax.get_legend_handles_labels()[1] +
           [f"C{c}: {', '.join(l.upper() for l in langs)}" for c, langs in CLUSTERS.items()],
    fontsize=8, loc='upper right', ncol=2
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_synergy_per_language_bar.png"), dpi=150)
plt.close()
print("→ 3_synergy_per_language_bar.png")


# =============================================================================
# PLOT 4 — Detail per ablation language: ΔCE top-1 / top-2 / top-1+2
#          one subplot per language, evaluated on itself (target)
# =============================================================================

n_cols = 5
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(5 * n_cols, 4 * n_rows),
                          gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
axes = axes.flatten()

for ax_idx, lang_ablate in enumerate(TARGET_LANGUAGES):
    ax      = axes[ax_idx]
    rec_tgt = next(r for r in records
                   if r["lang_ablate"] == lang_ablate and r["role"] == "target")

    d1  = rec_tgt["delta_top1"]
    d2  = rec_tgt["delta_top2"]
    d12 = rec_tgt["delta_top1p2"]
    add = d1 + d2

    ax.plot(LAYERS, d1,  color='#4C72B0', lw=1.6, ls='-',  marker='.', ms=4,
            label='Ablate top-1 only')
    ax.plot(LAYERS, d2,  color='#DD8452', lw=1.6, ls='--', marker='.', ms=4,
            label='Ablate top-2 only')
    ax.plot(LAYERS, add, color='#888888', lw=1.2, ls=':',
            label='Additive baseline')
    ax.plot(LAYERS, d12, color='#C44E52', lw=2.4, ls='-',  marker='o', ms=4,
            label='Ablate top-1+2')

    ax.fill_between(LAYERS, add, d12,
                    where=d12 >= add, alpha=0.22, color='#C44E52',
                    label='Positive synergy')
    ax.fill_between(LAYERS, add, d12,
                    where=d12 < add,  alpha=0.18, color='#4C72B0',
                    label='Negative synergy')

    ax.axhline(0, color='grey', linestyle=':', linewidth=0.7)
    ax.set_title(f"Ablate {LANG_NAMES[lang_ablate]}\n(eval on itself)",
                 fontsize=8.5, fontweight='bold',
                 color=CLUSTER_COLORS[LANG_TO_CLUSTER[lang_ablate]])
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("ΔCE", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=5, fontsize=7.5,
           framealpha=0.5, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(
    f"[{MODEL_LABEL}] Top-1 vs top-2 vs top-1+2 ablation — self-eval per language\n"
    "Red fill = super-additive synergy  |  Blue fill = sub-additive",
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig(os.path.join(OUTPUT_DIR, "4_ablation_detail_per_language.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("→ 4_ablation_detail_per_language.png")


# =============================================================================
# PLOT 5 — Selected pairs: intra vs inter cluster
#          Built dynamically from records to avoid KeyError
# =============================================================================

def pick_focus_pairs():
    pairs = []
    for la, le_role in [("fr", "target"), ("fr", "close"), ("fr", "distant"),
                         ("es", "target"), ("es", "close"), ("es", "distant")]:
        rec = next((r for r in records
                    if r["lang_ablate"] == la and r["role"] == le_role), None)
        if rec is None:
            continue
        le   = rec["lang_eval"]
        crel = rec["cluster_rel"]
        title = (f"{la.upper()}→{le.upper()} "
                 f"({'target' if le_role == 'target' else le_role}, "
                 f"{'intra C' + str(LANG_TO_CLUSTER[la]) if crel == 'intra' else 'inter'})")
        pairs.append((la, le, title, crel, rec))
    return pairs

FOCUS_PAIRS = pick_focus_pairs()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (la, le, title, crel, rec) in zip(axes, FOCUS_PAIRS):
    d1, d2, d12 = rec["delta_top1"], rec["delta_top2"], rec["delta_top1p2"]
    add = d1 + d2
    syn = d12 - add

    ax.plot(LAYERS, d1,  color='#4C72B0', lw=1.8, ls='-',  marker='.', ms=5,
            label=f'Ablate {LANG_NAMES[la]} top-1 only')
    ax.plot(LAYERS, d2,  color='#DD8452', lw=1.8, ls='--', marker='.', ms=5,
            label=f'Ablate {LANG_NAMES[la]} top-2 only')
    ax.plot(LAYERS, add, color='#888888', lw=1.2, ls=':',
            label='Additive sum (expected if independent)')
    ax.plot(LAYERS, d12, color='#C44E52', lw=2.4, ls='-',  marker='o', ms=5,
            label='Ablate both (top-1+2)')

    ax.fill_between(LAYERS, add, d12,
                    where=d12 >= add, alpha=0.25, color='#C44E52',
                    label='Positive synergy')
    ax.fill_between(LAYERS, add, d12,
                    where=d12 < add,  alpha=0.20, color='#4C72B0',
                    label='Negative synergy')

    border_c = '#C44E52' if crel == "intra" else '#4C72B0'
    ax.set_title(f"{title}\nmean synergy = {float(syn.mean()):+.4f}",
                 fontsize=9, fontweight='bold', color=border_c)
    ax.axhline(0, color='grey', ls=':', lw=0.7)
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel(f"ΔCE (eval: {LANG_NAMES[le]})", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(ls='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=4, fontsize=8.5,
           framealpha=0.5, bbox_to_anchor=(0.5, -0.03))
fig.suptitle(
    f"[{MODEL_LABEL}] Cluster synergy — FR & ES ablation: intra vs inter-cluster\n"
    "Red title = intra-cluster pair  |  Blue title = inter-cluster pair",
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "5_cluster_focus.png"), dpi=150)
plt.close()
print("→ 5_cluster_focus.png")


# =============================================================================
# Summary table
# =============================================================================

print(f"\n── [{MODEL_LABEL}] Summary: mean synergy by role & cluster relationship ──")
print(f"{'Role':<10} {'Cluster rel':<12} {'N pairs':>8} {'Mean syn':>10} {'Std':>8}")
print("─" * 52)
for role in ["target", "close", "distant"]:
    for crel in ["intra", "inter"]:
        sel = [r for r in records if r["role"] == role and r["cluster_rel"] == crel]
        if not sel:
            continue
        all_syn = np.concatenate([r["synergy"] for r in sel])
        print(f"{role:<10} {crel:<12} {len(sel):>8} {all_syn.mean():>+10.5f} {all_syn.std():>8.5f}")

print(f"\n✅ All plots saved to '{OUTPUT_DIR}/'  —  5 plots")
