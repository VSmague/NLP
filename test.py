"""
cluster_synergy_from_cache.py
==============================
Analyse la synergie intra-cluster vs inter-cluster
EN RÉUTILISANT le cache fig6_cache.pkl déjà calculé.

Pour chaque triplet (lang_ablate, lang_close, lang_distant) :
  fig6_data[lang_ablate]["top-1"  ][lang_eval]  = ΔCE ablate top-1 de lang_ablate
  fig6_data[lang_ablate]["top-2"  ][lang_eval]  = ΔCE ablate top-2 de lang_ablate
  fig6_data[lang_ablate]["top-1+2"][lang_eval]  = ΔCE ablate top-1+2 ensemble

Synergy = ΔCE_top1+2 − (ΔCE_top1 + ΔCE_top2)

Paires de proximité typologique utilisées lors du calcul :
  en  → close: fr,  distant: zh
  es  → close: pt,  distant: ja
  fr  → close: es,  distant: th
  ja  → close: ko,  distant: en
  ko  → close: ja,  distant: ar
  pt  → close: es,  distant: th
  th  → close: vi,  distant: ar
  vi  → close: th,  distant: en
  zh  → close: ja,  distant: es
  ar  → close: en,  distant: zh

Note : la synergie ici est INTRA-feature (top-1 vs top-2 de la MÊME langue),
       évaluée sur une langue cible, proche, ou distante.
       -> On s'attend à ce que la synergie soit forte sur la langue cible
          et faible sur les langues hors-cluster.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Parameters ────────────────────────────────────────────────────────────────
SAVE_DIR   = "sae_features"
OUTPUT_DIR = "plots_synergy_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FIG6 = os.path.join(SAVE_DIR, "fig6_cache.pkl")

# ── Clusters ──────────────────────────────────────────────────────────────────
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

# Paires de proximité (identiques à reproduce_figures_section5.py)
LANG_CLOSE = {
    'en': 'fr',  'es': 'pt',  'fr': 'es',  'ja': 'ko',  'ko': 'ja',
    'pt': 'es',  'th': 'vi',  'vi': 'th',  'zh': 'ja',  'ar': 'en',
}
LANG_DISTANT = {
    'en': 'zh',  'es': 'ja',  'fr': 'th',  'ja': 'en',  'ko': 'ar',
    'pt': 'th',  'th': 'ar',  'vi': 'en',  'zh': 'es',  'ar': 'zh',
}

FIG6_TRIPLES = {
    lang: (lang, LANG_CLOSE[lang], LANG_DISTANT[lang])
    for lang in TARGET_LANGUAGES
}


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


# =============================================================================
# Compute synergy arrays
# =============================================================================
# For each lang_ablate and each eval_lang in its triple:
#   synergy[layer] = ΔCE_top1+2[layer] - (ΔCE_top1[layer] + ΔCE_top2[layer])
#
# Each (lang_ablate, lang_eval) pair is tagged:
#   "target"  if lang_eval == lang_ablate
#   "close"   if lang_eval == LANG_CLOSE[lang_ablate]
#   "distant" if lang_eval == LANG_DISTANT[lang_ablate]
#
# Cluster relationship between lang_ablate and lang_eval:
#   "intra"  if same cluster
#   "inter"  if different cluster
# =============================================================================

records = []   # list of dicts with all info per (lang_ablate, lang_eval)

for lang_ablate in TARGET_LANGUAGES:
    lang_target, lang_close, lang_distant = FIG6_TRIPLES[lang_ablate]

    for role, lang_eval in [("target", lang_target),
                             ("close",  lang_close),
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
#           averaged over all 10 ablation languages
# =============================================================================

def avg_synergy_by_role(role):
    sel = [r for r in records if r["role"] == role]
    mat = np.stack([r["synergy"] for r in sel])   # shape (n_pairs, n_layers)
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
            linestyle=sty['ls'], marker=sty['marker'], markersize=4,
            label=sty['label'])
    ax.fill_between(LAYERS, mean - std, mean + std,
                    color=sty['color'], alpha=0.12)

ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Mean synergy  ΔCE_(1+2) − (ΔCE_1 + ΔCE_2)", fontsize=10)
ax.set_title("Feature synergy per layer — target vs close vs distant\n"
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
# PLOT 2 — Synergy per layer: intra-cluster vs inter-cluster
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
ax.set_title("Intra-cluster vs inter-cluster synergy per layer\n"
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
x      = np.arange(len(TARGET_LANGUAGES))
width  = 0.28

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
    ax.bar(x + offset, means, width, label=label, color=color, alpha=0.85,
           edgecolor='white')

ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax.set_xticks(x)
ax.set_xticklabels([LANG_NAMES[l] for l in TARGET_LANGUAGES], rotation=30,
                   ha='right', fontsize=9)
ax.set_ylabel("Mean synergy (avg over layers)", fontsize=10)
ax.set_title("Feature synergy per ablation language\n"
             "Target / Typologically close / Typologically distant",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_synergy_per_language_bar.png"), dpi=150)
plt.close()
print("→ 3_synergy_per_language_bar.png")


# =============================================================================
# PLOT 4 — Detail per ablation language: ΔCE top-1 / top-2 / top-1+2
#          one subplot per language, 3 curves (target eval only)
# =============================================================================

n_cols = 5
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                          gridspec_kw={'hspace': 0.55, 'wspace': 0.35})
axes = axes.flatten()

CONFIG_COLORS = {'top-1': '#4C72B0', 'top-2': '#DD8452', 'top-1+2': '#C44E52'}
CONFIG_STYLES = {'top-1': '-',       'top-2': '--',      'top-1+2': '-'}
CONFIG_WIDTHS = {'top-1': 1.6,       'top-2': 1.6,       'top-1+2': 2.4}
CONFIG_LABELS = {
    'top-1'  : 'Ablate top-1 only',
    'top-2'  : 'Ablate top-2 only',
    'top-1+2': 'Ablate top-1 & top-2',
}

for ax_idx, lang_ablate in enumerate(TARGET_LANGUAGES):
    ax      = axes[ax_idx]
    rec_tgt = next(r for r in records
                   if r["lang_ablate"] == lang_ablate and r["role"] == "target")

    d1   = rec_tgt["delta_top1"]
    d2   = rec_tgt["delta_top2"]
    d12  = rec_tgt["delta_top1p2"]
    add  = d1 + d2

    ax.plot(LAYERS, d1,  color=CONFIG_COLORS['top-1'],   linewidth=CONFIG_WIDTHS['top-1'],
            linestyle=CONFIG_STYLES['top-1'],   marker='.', markersize=4,
            label=CONFIG_LABELS['top-1'])
    ax.plot(LAYERS, d2,  color=CONFIG_COLORS['top-2'],   linewidth=CONFIG_WIDTHS['top-2'],
            linestyle=CONFIG_STYLES['top-2'],   marker='.', markersize=4,
            label=CONFIG_LABELS['top-2'])
    ax.plot(LAYERS, add, color='#888888', linewidth=1.2,
            linestyle=':', label='Sum (additive baseline)')
    ax.plot(LAYERS, d12, color=CONFIG_COLORS['top-1+2'], linewidth=CONFIG_WIDTHS['top-1+2'],
            linestyle=CONFIG_STYLES['top-1+2'], marker='o', markersize=4,
            label=CONFIG_LABELS['top-1+2'])

    # Shade synergy region
    ax.fill_between(LAYERS, add, d12,
                    where=d12 >= add, alpha=0.22,
                    color='#C44E52', label='Positive synergy')
    ax.fill_between(LAYERS, add, d12,
                    where=d12 < add,  alpha=0.18,
                    color='#4C72B0', label='Negative synergy')

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
    "Top-1 vs top-2 vs top-1+2 ablation — self-eval per language\n"
    "Red fill = super-additive synergy | Blue fill = sub-additive",
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig(os.path.join(OUTPUT_DIR, "4_ablation_detail_per_language.png"), dpi=150)
plt.close()
print("→ 4_ablation_detail_per_language.png")


# =============================================================================
# PLOT 5 — Cluster 3 focus: FR ablation → eval on ES (close/intra)
#          vs eval on TH (distant/inter)  — ΔCE + synergy side by side
# =============================================================================

# Pick representative intra pairs from cluster 3
# Build available focus pairs dynamically from records
# Pick: for FR and ES → target + close + one inter pair each
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
                 f"({'target' if le_role=='target' else le_role}, "
                 f"{'intra C'+str(LANG_TO_CLUSTER[la]) if crel=='intra' else 'inter'})")
        pairs.append((la, le, title, crel, rec))
    return pairs

FOCUS_PAIRS = pick_focus_pairs()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (la, le, title, crel, rec) in zip(axes, FOCUS_PAIRS):
    d1, d2, d12 = rec["delta_top1"], rec["delta_top2"], rec["delta_top1p2"]

    add = d1 + d2
    syn = d12 - add

    ax.plot(LAYERS, d1,  color='#4C72B0', lw=1.8, ls='-',  marker='.', ms=4,
            label='Ablate top-1 only')
    ax.plot(LAYERS, d2,  color='#DD8452', lw=1.8, ls='--', marker='.', ms=4,
            label='Ablate top-2 only')
    ax.plot(LAYERS, add, color='#888888', lw=1.2, ls=':',
            label='Additive sum')
    ax.plot(LAYERS, d12, color='#C44E52', lw=2.4, ls='-',  marker='o', ms=4,
            label='Ablate both (top-1+2)')
    ax.fill_between(LAYERS, add, d12,
                    where=d12 >= add, alpha=0.25, color='#C44E52')
    ax.fill_between(LAYERS, add, d12,
                    where=d12 < add,  alpha=0.20, color='#4C72B0')

    mean_syn = float(syn.mean())
    border_c = '#C44E52' if crel == "intra" else '#4C72B0'
    ax.set_title(f"{title}\nmean synergy = {mean_syn:+.4f}",
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
    "Cluster 3 focus — FR & ES ablation: intra-cluster vs inter-cluster synergy\n"
    "Red title = intra-cluster pair  |  Blue title = inter-cluster pair",
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "5_cluster3_focus.png"), dpi=150)
plt.close()
print("→ 5_cluster3_focus.png")


# =============================================================================
# Summary table
# =============================================================================

print("\n── Summary: mean synergy by role & cluster relationship ──")
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