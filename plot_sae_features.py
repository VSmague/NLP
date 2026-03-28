"""
plot_sae_features.py
====================
Visualisations des features SAE sauvegardées dans sae_features/
Lancer avec : python plot_sae_features.py
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR   = "sae_features"
OUTPUT_DIR = "output/plots_sae_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANG_COLORS = {
    'en': '#4C72B0', 'es': '#DD8452', 'fr': '#55A868',
    'ja': '#C44E52', 'ko': '#8172B3', 'pt': '#937860',
    'th': '#DA8BC3', 'vi': '#8C8C8C', 'zh': '#CCB974', 'ar': '#64B5CD',
}
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese',
    'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic',
}

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
})

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement des données...")

with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)

LAYERS    = meta["layers"]
LANGUAGES = meta["languages"]
D_SAE     = meta["d_sae"]

top_index  = {}
top_values = {}
readable   = {}

for layer in LAYERS:
    top_index[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"), weights_only=True
    )
    top_values[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_values.pt"), weights_only=True
    )
    with open(os.path.join(SAVE_DIR, f"layer_{layer}_readable.json")) as f:
        readable[layer] = json.load(f)

print(f"  {len(LAYERS)} layers × {len(LANGUAGES)} langues × {D_SAE} features\n")


# =============================================================================
# PLOT 1 — Top-3 v-scores par langue et par layer (heatmap)
# =============================================================================
print("[1] Heatmap top-1 v-score par langue × layer...")

data_heatmap = np.zeros((len(LANGUAGES), len(LAYERS)))
for j, layer in enumerate(LAYERS):
    for i, lang in enumerate(LANGUAGES):
        data_heatmap[i, j] = readable[layer][lang][0]["v_score"]

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(data_heatmap, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(LAYERS)))
ax.set_xticklabels([f"L{l}" for l in LAYERS])
ax.set_yticks(range(len(LANGUAGES)))
ax.set_yticklabels([LANG_NAMES[l] for l in LANGUAGES])
for i in range(len(LANGUAGES)):
    for j in range(len(LAYERS)):
        ax.text(j, i, f"{data_heatmap[i,j]:.3f}", ha='center', va='center',
                fontsize=7.5, color='black' if data_heatmap[i,j] < 0.15 else 'white')
plt.colorbar(im, ax=ax, label="v-score (top-1 feature)")
ax.set_title("Top-1 v-score par langue et par couche", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_heatmap_vscore.png"))
plt.close()
print("   → 1_heatmap_vscore.png")


# =============================================================================
# PLOT 2 — Évolution du v-score top-1 par langue à travers les layers (line)
# =============================================================================
print("[2] Évolution du v-score top-1 par langue...")

fig, ax = plt.subplots(figsize=(11, 5))
for lang in LANGUAGES:
    scores = [readable[layer][lang][0]["v_score"] for layer in LAYERS]
    ax.plot(LAYERS, scores, marker='o', label=LANG_NAMES[lang],
            color=LANG_COLORS[lang], linewidth=2, markersize=5)
ax.set_xlabel("Couche", fontsize=11)
ax.set_ylabel("v-score (top-1 feature)", fontsize=11)
ax.set_title("Spécificité linguistique à travers les couches", fontsize=13, fontweight='bold')
ax.legend(ncol=2, fontsize=9, framealpha=0.4)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_vscore_evolution.png"))
plt.close()
print("   → 2_vscore_evolution.png")


# =============================================================================
# PLOT 3 — Distribution des v-scores (top-20) pour chaque langue à layer 18
# =============================================================================
print("[3] Distribution des top-20 v-scores (layer 18)...")

TARGET_LAYER = 18 if 18 in LAYERS else LAYERS[len(LAYERS)//2]

fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=False)
axes = axes.flatten()

for idx, lang in enumerate(LANGUAGES):
    scores = [readable[TARGET_LAYER][lang][k]["v_score"] for k in range(20)]
    ranks  = [f"#{k+1}" for k in range(20)]
    axes[idx].bar(range(20), scores, color=LANG_COLORS[lang], alpha=0.85, width=0.7)
    axes[idx].set_title(LANG_NAMES[lang], fontweight='bold', fontsize=10)
    axes[idx].set_xticks([0, 4, 9, 14, 19])
    axes[idx].set_xticklabels(['#1','#5','#10','#15','#20'], fontsize=8)
    axes[idx].set_ylabel("v-score" if idx % 5 == 0 else "", fontsize=8)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.4)

fig.suptitle(f"Distribution des v-scores top-20 par langue (Layer {TARGET_LAYER})",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_vscore_distribution_top20.png"), bbox_inches='tight')
plt.close()
print("   → 3_vscore_distribution_top20.png")


# =============================================================================
# PLOT 4 — Overlap des top-K features entre langues (matrice de similarité)
# =============================================================================
print("[4] Overlap des top-K features entre langues...")

def feature_overlap(layer, k=50):
    sets = {}
    for lang in LANGUAGES:
        sets[lang] = set(top_index[layer][LANGUAGES.index(lang), :k].tolist())
    matrix = np.zeros((len(LANGUAGES), len(LANGUAGES)))
    for i, la in enumerate(LANGUAGES):
        for j, lb in enumerate(LANGUAGES):
            inter = len(sets[la] & sets[lb])
            union = len(sets[la] | sets[lb])
            matrix[i, j] = inter / union if union > 0 else 0.0
    return matrix

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (layer, k) in zip(axes, [(LAYERS[0], 50), (TARGET_LAYER, 50), (LAYERS[-1], 50)]):
    mat = feature_overlap(layer, k)
    mask = np.eye(len(LANGUAGES), dtype=bool)
    sns.heatmap(mat, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[l.upper() for l in LANGUAGES],
                yticklabels=[l.upper() for l in LANGUAGES],
                mask=mask, vmin=0, vmax=0.3,
                annot_kws={"size": 8}, linewidths=0.3)
    ax.set_title(f"Layer {layer} — Jaccard top-{k}", fontweight='bold', fontsize=10)

fig.suptitle("Overlap des top-50 features entre langues (Jaccard index)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_overlap_jaccard.png"))
plt.close()
print("   → 4_feature_overlap_jaccard.png")


# =============================================================================
# PLOT 5 — Chute du v-score : top-1 vs top-2 vs top-3 (gap)
# =============================================================================
print("[5] Chute du v-score entre top-1, top-2, top-3...")

fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=False)
axes = axes.flatten()

for idx, lang in enumerate(LANGUAGES):
    for layer in LAYERS:
        scores_top3 = [readable[layer][lang][k]["v_score"] for k in range(3)]
        axes[idx].plot(["#1","#2","#3"], scores_top3,
                       marker='o', alpha=0.6, linewidth=1.5,
                       label=f"L{layer}", color=plt.cm.plasma(layer / max(LAYERS)))
    axes[idx].set_title(LANG_NAMES[lang], fontweight='bold', fontsize=10)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.4)
    axes[idx].set_ylabel("v-score" if idx % 5 == 0 else "")

# Légende commune
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(LAYERS),
           fontsize=8, framealpha=0.4, title="Layer")
fig.suptitle("Chute du v-score : top-1 → top-2 → top-3 par couche",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(os.path.join(OUTPUT_DIR, "5_vscore_dropoff.png"))
plt.close()
print("   → 5_vscore_dropoff.png")


# =============================================================================
# PLOT 6 — Quelle langue a les features les plus spécialisées ? (bar comparatif)
# =============================================================================
print("[6] Comparaison de la spécificité moyenne par langue...")

rows = []
for layer in LAYERS:
    for lang in LANGUAGES:
        top3_mean = np.mean([readable[layer][lang][k]["v_score"] for k in range(3)])
        top1      = readable[layer][lang][0]["v_score"]
        rows.append({"Layer": f"L{layer}", "Language": LANG_NAMES[lang],
                     "top1_vscore": top1, "top3_mean": top3_mean})
df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Moyenne toutes layers — top-1
mean_top1 = df.groupby("Language")["top1_vscore"].mean().sort_values(ascending=False)
bars = axes[0].bar(mean_top1.index, mean_top1.values,
                   color=[LANG_COLORS[k] for k in [
                       l for l, n in LANG_NAMES.items() if n in mean_top1.index
                   ]], alpha=0.85)
axes[0].set_title("v-score moyen (top-1) toutes couches", fontweight='bold')
axes[0].set_ylabel("v-score moyen")
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
for bar, val in zip(bars, mean_top1.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=8)

# Par layer — grouped
pivot = df.pivot(index="Language", columns="Layer", values="top1_vscore")
pivot.plot(kind='bar', ax=axes[1], colormap='tab10', alpha=0.85, width=0.8)
axes[1].set_title("v-score top-1 par couche et par langue", fontweight='bold')
axes[1].set_ylabel("v-score")
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend(title="Layer", fontsize=8, ncol=2)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle("Quelle langue a les features les plus spécialisées ?",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "6_language_specialization.png"))
plt.close()
print("   → 6_language_specialization.png")


# =============================================================================
# PLOT 7 — Unicité des features : combien de features top-K sont partagées
#           avec au moins une autre langue ?
# =============================================================================
print("[7] Unicité des features top-K par langue...")

Ks = [1, 5, 10, 20, 50]

fig, axes = plt.subplots(1, len(LAYERS), figsize=(18, 4), sharey=True)

for ax, layer in zip(axes, LAYERS):
    uniqueness = {lang: [] for lang in LANGUAGES}
    for k in Ks:
        all_sets = {lang: set(top_index[layer][LANGUAGES.index(lang), :k].tolist())
                    for lang in LANGUAGES}
        for lang in LANGUAGES:
            shared = sum(
                len(all_sets[lang] & all_sets[other]) > 0
                for other in LANGUAGES if other != lang
            )
            # Proportion de features NON partagées
            own   = all_sets[lang]
            others_union = set().union(*[all_sets[o] for o in LANGUAGES if o != lang])
            unique_ratio = len(own - others_union) / len(own) if own else 0
            uniqueness[lang].append(unique_ratio)

    for lang in LANGUAGES:
        ax.plot(Ks, uniqueness[lang], marker='o', label=lang.upper(),
                color=LANG_COLORS[lang], linewidth=2, markersize=4)
    ax.set_title(f"Layer {layer}", fontsize=9, fontweight='bold')
    ax.set_xlabel("Top-K", fontsize=8)
    ax.set_xticks(Ks)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Ratio de features uniques", fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10, fontsize=8,
           framealpha=0.4)
fig.suptitle("Unicité des top-K features par langue (pas partagées avec d'autres langues)",
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(os.path.join(OUTPUT_DIR, "7_feature_uniqueness.png"))
plt.close()
print("   → 7_feature_uniqueness.png")


# =============================================================================
# PLOT 8 — Scatter : feature_idx vs v_score pour toutes les langues (layer 18)
#           Montre la distribution dans l'espace des features
# =============================================================================
print("[8] Scatter feature_idx vs v-score (layer 18)...")

fig, ax = plt.subplots(figsize=(13, 5))

for lang in LANGUAGES:
    lang_idx = LANGUAGES.index(lang)
    # Top-100 features pour ce plot
    top100_idx = top_index[TARGET_LAYER][lang_idx, :100].numpy()
    top100_val = top_values[TARGET_LAYER][lang_idx, :100].numpy()
    ax.scatter(top100_idx, top100_val, alpha=0.5, s=18,
               color=LANG_COLORS[lang], label=LANG_NAMES[lang])

ax.set_xlabel("Indice de la feature (0 → 163840)", fontsize=11)
ax.set_ylabel("v-score", fontsize=11)
ax.set_title(f"Top-100 features par langue dans l'espace SAE — Layer {TARGET_LAYER}",
             fontsize=13, fontweight='bold')
ax.legend(ncol=2, fontsize=8, framealpha=0.4)
ax.grid(linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "8_scatter_feature_space.png"))
plt.close()
print("   → 8_scatter_feature_space.png")


# =============================================================================
# PLOT 9 — Résumé compact : top-1 feature index par langue et par layer
# =============================================================================
print("[9] Table résumé top-1 feature index...")

rows_table = []
for lang in LANGUAGES:
    row = {"Langue": LANG_NAMES[lang]}
    for layer in LAYERS:
        feat  = readable[layer][lang][0]["feature_idx"]
        vscore = readable[layer][lang][0]["v_score"]
        row[f"L{layer}"] = f"{feat}\n({vscore:.3f})"
    rows_table.append(row)

df_table = pd.DataFrame(rows_table).set_index("Langue")

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
tbl = ax.table(
    cellText=df_table.values,
    rowLabels=df_table.index,
    colLabels=df_table.columns,
    cellLoc='center', loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.3, 2.2)

for (row, col), cell in tbl.get_celld().items():
    if row == 0 or col == -1:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#F0F4F8')
    cell.set_edgecolor('#CCCCCC')

ax.set_title("Top-1 feature index (et v-score) par langue × couche",
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "9_summary_table.png"), bbox_inches='tight')
plt.close()
print("   → 9_summary_table.png")


# =============================================================================
# FIN
# =============================================================================
print(f"\n✅ Tous les plots sauvegardés dans '{OUTPUT_DIR}/'")
print(f"   9 fichiers PNG générés")