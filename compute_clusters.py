"""
cluster_from_matrix.py
=======================
Clusterise les langues depuis ta matrice CSV (layer 18).
Lance avec : python cluster_from_matrix.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

# ── Charger la matrice ────────────────────────────────────────────────────────

df = pd.read_csv("plots_interaction/interaction_matrix_avg_layers.csv", index_col=0)
M  = df.values
LANGUAGES = list(df.index)
n = len(LANGUAGES)

TYPO_FAMILIES = {
    'en': 'Indo-European (Germanic)',
    'es': 'Indo-European (Romance)',
    'fr': 'Indo-European (Romance)',
    'pt': 'Indo-European (Romance)',
    'ar': 'Afro-Asiatic (Semitic)',
    'zh': 'Sino-Tibetan',
    'ja': 'Japonic',
    'ko': 'Koreanic',
    'vi': 'Austroasiatic',
    'th': 'Kra-Dai',
}
FAMILY_COLORS = {
    'Indo-European (Germanic)' : '#4C72B0',
    'Indo-European (Romance)'  : '#55A868',
    'Afro-Asiatic (Semitic)'   : '#C44E52',
    'Sino-Tibetan'             : '#CCB974',
    'Japonic'                  : '#DD8452',
    'Koreanic'                 : '#8172B3',
    'Austroasiatic'            : '#DA8BC3',
    'Kra-Dai'                  : '#64B5CD',
}

lang_labels = [l.upper() for l in LANGUAGES]

# =============================================================================
# Construire 3 matrices de distance
# =============================================================================

# Option 1 — distance euclidienne entre profils de lignes
D_row = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D_row[i, j] = np.linalg.norm(M[i] - M[j])
D_row = D_row / D_row.max()

# Option 2 — influence mutuelle symétrisée (recommandée)
D_sym = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        num   = M[i, j] + M[j, i]
        denom = M[i, i] + M[j, j]
        D_sym[i, j] = 1.0 - (num / denom) if denom > 1e-8 else 1.0
np.fill_diagonal(D_sym, 0)
D_sym = np.clip((D_sym + D_sym.T) / 2, 0, None)

# Option 3 — distance sur colonnes normalisées (impact relatif)
diag     = np.diag(M).copy()
diag[diag == 0] = 1e-8
M_norm   = M / diag[:, None]           # chaque ligne / auto-impact
D_norm   = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D_norm[i, j] = np.linalg.norm(M_norm[:, i] - M_norm[:, j])
D_norm = D_norm / D_norm.max()

DISTANCES = {
    "Option 1 — profil de lignes"      : D_row,
    "Option 2 — influence mutuelle ★"  : D_sym,
    "Option 3 — colonnes normalisées"  : D_norm,
}

# =============================================================================
# Plot 1 — Comparaison des 3 dendrogrammes
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (title, D) in zip(axes, DISTANCES.items()):
    Z  = linkage(squareform(D), method='ward')
    dn = dendrogram(Z, labels=lang_labels, ax=ax,
                    leaf_font_size=12, color_threshold=0.4 * max(Z[:, 2]))
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.set_ylabel("Distance Ward")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colorier les labels selon la famille typologique
    for lbl in ax.get_xmajorticklabels():
        lang = lbl.get_text().lower()
        lbl.set_color(FAMILY_COLORS.get(TYPO_FAMILIES.get(lang, ''), '#333333'))
        lbl.set_fontweight('bold')

legend_patches = [
    mpatches.Patch(color=col, label=fam)
    for fam, col in FAMILY_COLORS.items()
    if fam in [TYPO_FAMILIES[l] for l in LANGUAGES]
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=8, framealpha=0.4, bbox_to_anchor=(0.5, -0.05),
           title="Famille typologique (couleur des labels)")

fig.suptitle("Clustering des langues — comparaison des 3 méthodes de distance",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig("clustering_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ clustering_comparison.png")

# =============================================================================
# Plot 2 — Dendrogramme détaillé Option 2 + matrice réordonnée
# =============================================================================

D_best = D_sym
Z_best = linkage(squareform(D_best), method='ward')
order  = dendrogram(Z_best, no_plot=True)["leaves"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                          gridspec_kw={'width_ratios': [1.6, 1]})

# Dendrogramme
dn = dendrogram(Z_best, labels=lang_labels, ax=axes[0],
                leaf_font_size=13, color_threshold=0.4 * max(Z_best[:, 2]))
axes[0].set_title("Clustering hiérarchique Ward\n(distance = influence mutuelle)",
                  fontsize=11, fontweight='bold')
axes[0].set_ylabel("Distance Ward")
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

for lbl in axes[0].get_xmajorticklabels():
    lang = lbl.get_text().lower()
    lbl.set_color(FAMILY_COLORS.get(TYPO_FAMILIES.get(lang, ''), '#333'))
    lbl.set_fontweight('bold')

axes[0].legend(handles=legend_patches, fontsize=7.5, loc='upper right',
               title="Famille typologique", title_fontsize=8)

# Matrice réordonnée
M_r    = M[np.ix_(order, order)]
labs_r = [lang_labels[k] for k in order]
im     = axes[1].imshow(M_r, cmap='RdYlGn', aspect='auto')
axes[1].set_xticks(range(n))
axes[1].set_yticks(range(n))
axes[1].set_xticklabels(labs_r, fontsize=10, rotation=45)
axes[1].set_yticklabels(labs_r, fontsize=10)
axes[1].set_title("M réordonnée\n(blocs = langues proches)", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=axes[1], label="ΔCE")
for i in range(n):
    for j in range(n):
        axes[1].text(j, i, f"{M_r[i,j]:.2f}", ha='center', va='center',
                     fontsize=6.5, color='black' if M_r[i,j] < M.max()*0.6 else 'white')

plt.tight_layout()
plt.savefig("clustering_best.png", dpi=150, bbox_inches='tight')
plt.close()
print("→ clustering_best.png")

# =============================================================================
# Plot 3 — MDS 2D (carte des langues dans l'espace d'influence)
# =============================================================================

mds    = MDS(n_components=2, dissimilarity='precomputed',
             random_state=42, normalized_stress='auto')
coords = mds.fit_transform(D_best)

# Clusters automatiques (3 clusters)
clusters = fcluster(Z_best, t=5, criterion='maxclust')

fig, ax = plt.subplots(figsize=(8, 7))

for i, lang in enumerate(LANGUAGES):
    family = TYPO_FAMILIES.get(lang, '')
    color  = FAMILY_COLORS.get(family, '#888')
    ax.scatter(coords[i, 0], coords[i, 1], s=200, color=color,
               zorder=3, edgecolors='white', linewidths=1.5)
    ax.text(coords[i, 0] + 0.01, coords[i, 1] + 0.01,
            lang.upper(), fontsize=11, fontweight='bold', color=color)

# Cercles de cluster
for c in np.unique(clusters):
    idx    = np.where(clusters == c)[0]
    cx, cy = coords[idx, 0].mean(), coords[idx, 1].mean()
    radius = max(np.linalg.norm(coords[idx] - [cx, cy], axis=1)) + 0.08
    circle = plt.Circle((cx, cy), radius, fill=False,
                         linestyle='--', linewidth=1.2, color='grey', alpha=0.5)
    ax.add_patch(circle)
    ax.text(cx, cy - radius - 0.03, f"Cluster {c}",
            ha='center', fontsize=9, color='grey', style='italic')

ax.set_title("Carte MDS des langues — espace d'influence mutuelle\n"
             "(proximité = profil d'interaction similaire)",
             fontsize=12, fontweight='bold')
ax.legend(handles=legend_patches, fontsize=8, loc='upper right',
          title="Famille typologique")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.grid(linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("clustering_mds.png", dpi=150)
plt.close()
print("→ clustering_mds.png")

# =============================================================================
# Afficher les clusters trouvés
# =============================================================================

print("\n── Clusters détectés (Option 2, Ward, k=3) ──")
for c in sorted(np.unique(clusters)):
    langs_in = [LANGUAGES[i] for i in range(n) if clusters[i] == c]
    fams     = set(TYPO_FAMILIES[l] for l in langs_in)
    print(f"  Cluster {c} : {[l.upper() for l in langs_in]}")
    print(f"            ← {fams}")

print("\n3 fichiers générés : clustering_comparison.png  clustering_best.png  clustering_mds.png")