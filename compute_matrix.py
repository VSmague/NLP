"""
cross_linguistic_interaction.py
================================
Construit la matrice d'interaction cross-linguistique :

    M[i, j] = moyenne sur toutes les layers du delta-CE
              sur texte de langue i après ablation des top
              features de langue j à chaque layer

Prérequis : avoir tourné ablation.py au moins une fois
            (sae_features/ doit exister avec les .pt et .json)
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
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# ── Auth ──────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("secrets.ini")
os.environ["HF_TOKEN"] = config["huggingface"]["token"]

# ── Paramètres ────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-0.6B"
SAE_RELEASE = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR    = "sae_features"
OUTPUT_DIR  = "plots_interaction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERACTION_CACHE = os.path.join(SAVE_DIR, "interaction_matrix_avg_layers.pkl")

TOP_K_ABLATE = 2   # top-k features de langue j à ablater par layer
N_SENTENCES  = 20  # phrases par langue pour estimer le delta-CE

RECOMPUTE = True  # mettre True pour forcer le recalcul

# Familles typologiques
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

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================================================
# Chargement du cache SAE
# =============================================================================

print("\n[1] Chargement du cache SAE...")
with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)

LANGUAGES = meta["languages"]
LAYERS    = meta["layers"]
n_langs   = len(LANGUAGES)
n_layers  = len(LAYERS)

top_index_all = {}
for layer in LAYERS:
    top_index_all[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"),
        weights_only=True
    )

print(f"   {n_layers} layers × {n_langs} langues")
print(f"   Layers : {LAYERS}")
print(f"   Estimation durée : ~{n_langs * n_langs * n_layers * N_SENTENCES // 60} min sur GPU\n")


# =============================================================================
# Fonctions utilitaires
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
        else:
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


# =============================================================================
# Construction de M[i, j] — boucle optimisée : un SAE chargé une fois par
# (layer, lang_ablate), évalué sur toutes les lang_eval
# =============================================================================

if RECOMPUTE or not os.path.exists(INTERACTION_CACHE):

    print("[2] Chargement modèle et données...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()

    import requests
    DATA_URL = ("https://raw.githubusercontent.com/Aatrox103/"
                "multilingual-llm-features/main/SAE/data/multilingual_data.jsonl")
    resp = requests.get(DATA_URL)
    resp.raise_for_status()
    multilingual_texts = []
    for line in resp.text.strip().split('\n'):
        obj = json.loads(line)
        multilingual_texts.append(obj['text'] if isinstance(obj, dict) else obj)
    print(f"   {len(multilingual_texts)} phrases chargées")

    # Pré-tokeniser toutes les phrases pour éviter de répéter le tokenizer
    sentences_per_lang = {
        i: [t for t in multilingual_texts[i*100 : i*100 + N_SENTENCES]]
        for i in range(n_langs)
    }

    print(f"\n[3] Construction de M[i,j] — moyenne sur {n_layers} layers...")
    print(f"   Stratégie : charge chaque SAE une fois → évalue toutes les langues")
    print(f"   {n_langs}×{n_layers} chargements SAE au total\n")

    # M_per_layer[i, j, layer_idx] = delta-CE texte_i après ablation features_j à layer
    M_per_layer = np.zeros((n_langs, n_langs, n_layers))
    t0 = time.time()

    for j, lang_ablate in enumerate(LANGUAGES):
        for li, layer in enumerate(LAYERS):

            sae      = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)
            feat_idx = top_index_all[layer][j, :TOP_K_ABLATE].to(device)

            for i, lang_eval in enumerate(LANGUAGES):
                deltas = []
                for text in sentences_per_lang[i]:
                    inp = tokenizer.encode(text, return_tensors='pt',
                                           add_special_tokens=True).to(device)
                    if inp.size(1) < 2:
                        continue
                    base_ce = compute_ce_loss(model.forward(inp)["logits"], inp).item()
                    abl_ce  = compute_ce_loss(
                        directional_ablation_forward(model, sae, layer, feat_idx, inp)["logits"],
                        inp
                    ).item()
                    deltas.append(abl_ce - base_ce)
                M_per_layer[i, j, li] = float(np.nanmean(deltas)) if deltas else 0.0

            del sae

            # Progression
            done  = j * n_layers + li + 1
            total = n_langs * n_layers
            pct   = done / total * 100
            eta   = ((time.time() - t0) / done) * (total - done) if done > 0 else 0
            print(f"   [{lang_ablate.upper()} | L{layer:2d}] "
                  f"{pct:5.1f}% — ETA {eta/60:.1f} min", flush=True)

    # Moyenne sur les layers
    M = M_per_layer.mean(axis=2)

    with open(INTERACTION_CACHE, "wb") as f:
        pickle.dump({
            "M"           : M,
            "M_per_layer" : M_per_layer,
            "languages"   : LANGUAGES,
            "layers"      : LAYERS,
            "top_k"       : TOP_K_ABLATE,
            "method"      : "avg_layers",
        }, f)
    print(f"\n✅ Matrice sauvegardée dans {INTERACTION_CACHE}")

else:
    print(f"\n[2-3] Chargement depuis le cache ({INTERACTION_CACHE})...")
    with open(INTERACTION_CACHE, "rb") as f:
        cache = pickle.load(f)
    M           = cache["M"]
    M_per_layer = cache.get("M_per_layer", None)
    LANGUAGES   = cache["languages"]
    LAYERS      = cache["layers"]
    n_langs     = len(LANGUAGES)
    n_layers    = len(LAYERS)
    print(f"   ✅ M shape: {M.shape}  layers={LAYERS}  top_k={cache['top_k']}")


# =============================================================================
# PLOTS
# =============================================================================

lang_labels = [l.upper() for l in LANGUAGES]

# ── Plot 1 — Matrice brute ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(M, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(n_langs))
ax.set_yticks(range(n_langs))
ax.set_xticklabels(lang_labels, fontsize=11)
ax.set_yticklabels(lang_labels, fontsize=11)
ax.set_xlabel("Langue dont on ablate les features (j)", fontsize=11)
ax.set_ylabel("Langue évaluée (i)", fontsize=11)
for i in range(n_langs):
    for j in range(n_langs):
        ax.text(j, i, f"{M[i,j]:+.3f}", ha='center', va='center', fontsize=8,
                color='black' if abs(M[i,j]) < M.max()*0.6 else 'white')
plt.colorbar(im, ax=ax, label=f"ΔCE moyen (moy. {n_layers} layers)")
ax.set_title(f"Matrice d'interaction cross-linguistique\n"
             f"M[i,j] = ΔCE moyen texte i | ablation features j "
             f"(top-{TOP_K_ABLATE}, moy. {n_layers} layers)",
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_interaction_matrix_raw.png"), dpi=150)
plt.close()
print("→ 1_interaction_matrix_raw.png")

# ── Plot 2 — Matrice normalisée ───────────────────────────────────────────────
diag = np.diag(M).copy()
diag[diag == 0] = 1e-8
M_norm = M / diag[:, None]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(M_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(n_langs))
ax.set_yticks(range(n_langs))
ax.set_xticklabels(lang_labels, fontsize=11)
ax.set_yticklabels(lang_labels, fontsize=11)
ax.set_xlabel("Langue dont on ablate les features (j)", fontsize=11)
ax.set_ylabel("Langue évaluée (i)", fontsize=11)
for i in range(n_langs):
    for j in range(n_langs):
        ax.text(j, i, f"{M_norm[i,j]:.2f}", ha='center', va='center', fontsize=8.5,
                color='black' if M_norm[i,j] < 0.6 else 'white')
plt.colorbar(im, ax=ax, label="ΔCE(i,j) / ΔCE(i,i)")
for k in range(n_langs):
    ax.add_patch(plt.Rectangle((k-0.5, k-0.5), 1, 1,
                                fill=False, edgecolor='black', lw=2))
ax.set_title("Matrice normalisée — impact relatif à l'auto-ablation",
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_interaction_matrix_normalized.png"), dpi=150)
plt.close()
print("→ 2_interaction_matrix_normalized.png")

# ── Plot 3 — Clustering hiérarchique ─────────────────────────────────────────
D = np.zeros((n_langs, n_langs))
for i in range(n_langs):
    for j in range(n_langs):
        num   = M[i, j] + M[j, i]
        denom = M[i, i] + M[j, j]
        D[i, j] = 1.0 - (num / denom) if denom > 1e-8 else 1.0
np.fill_diagonal(D, 0)
D = np.clip((D + D.T) / 2, 0, None)
Z = linkage(squareform(D), method='ward')

legend_patches = [
    mpatches.Patch(color=col, label=fam)
    for fam, col in FAMILY_COLORS.items()
    if fam in [TYPO_FAMILIES[l] for l in LANGUAGES]
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                          gridspec_kw={'width_ratios': [2, 1]})
dendrogram(Z, labels=lang_labels, ax=axes[0],
           leaf_font_size=12, color_threshold=0.5*max(Z[:,2]))
axes[0].set_title("Clustering hiérarchique (Ward) — moy. layers",
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel("Distance Ward")
axes[0].legend(handles=legend_patches, fontsize=7.5, loc='upper right',
               title="Famille typologique", title_fontsize=8)

order       = dendrogram(Z, no_plot=True)["leaves"]
M_reordered = M[np.ix_(order, order)]
labs_r      = [lang_labels[k] for k in order]
im = axes[1].imshow(M_reordered, cmap='RdYlGn', aspect='auto')
axes[1].set_xticks(range(n_langs))
axes[1].set_yticks(range(n_langs))
axes[1].set_xticklabels(labs_r, fontsize=10, rotation=45)
axes[1].set_yticklabels(labs_r, fontsize=10)
axes[1].set_title("M réordonnée", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=axes[1], label="ΔCE moyen")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_hierarchical_clustering.png"), dpi=150)
plt.close()
print("→ 3_hierarchical_clustering.png")

# ── Plot 4 — Clustermap seaborn ──────────────────────────────────────────────
df_M = pd.DataFrame(M, index=lang_labels, columns=lang_labels)
rc   = pd.Series({l.upper(): FAMILY_COLORS[TYPO_FAMILIES[l]] for l in LANGUAGES})

g = sns.clustermap(df_M, cmap='RdYlGn', figsize=(10, 10),
                   annot=True, fmt=".3f", annot_kws={"size": 8},
                   row_colors=rc, col_colors=rc,
                   dendrogram_ratio=0.15,
                   cbar_pos=(0.02, 0.8, 0.03, 0.15),
                   linewidths=0.5)
g.ax_heatmap.set_xlabel("Langue ablated (j)", fontsize=10)
g.ax_heatmap.set_ylabel("Langue évaluée (i)", fontsize=10)
g.fig.suptitle("Clustermap — moyenne sur toutes les layers",
               fontsize=11, fontweight='bold', y=1.01)
g.fig.legend(handles=legend_patches, loc='lower left', fontsize=8,
             title="Famille typologique", title_fontsize=9,
             bbox_to_anchor=(0.0, 0.0))
plt.savefig(os.path.join(OUTPUT_DIR, "4_clustermap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("→ 4_clustermap.png")

# ── Plot 5 — Super-language clusters ─────────────────────────────────────────
cluster_labels_arr = fcluster(Z, t=3, criterion='maxclust')
clusters = {}
for i, lang in enumerate(LANGUAGES):
    clusters.setdefault(int(cluster_labels_arr[i]), []).append(lang)

print("\nClusters détectés :")
for c, langs in sorted(clusters.items()):
    print(f"  Cluster {c}: {[l.upper() for l in langs]}"
          f"  ← {set(TYPO_FAMILIES[l] for l in langs)}")

cluster_ids = sorted(clusters.keys())
n_clusters  = len(cluster_ids)
M_cluster   = np.zeros((n_clusters, n_clusters))
for ci, c_eval in enumerate(cluster_ids):
    for cj, c_ablate in enumerate(cluster_ids):
        idxs_eval   = [LANGUAGES.index(l) for l in clusters[c_eval]]
        idxs_ablate = [LANGUAGES.index(l) for l in clusters[c_ablate]]
        M_cluster[ci, cj] = np.mean([M[i, j] for i in idxs_eval for j in idxs_ablate])

cluster_names = [
    f"C{c}:\n{chr(10).join(l.upper() for l in clusters[c])}"
    for c in cluster_ids
]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(M_cluster, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(n_clusters))
ax.set_yticks(range(n_clusters))
ax.set_xticklabels(cluster_names, fontsize=9)
ax.set_yticklabels(cluster_names, fontsize=9)
ax.set_xlabel("Cluster ablated", fontsize=10)
ax.set_ylabel("Cluster évalué", fontsize=10)
for i in range(n_clusters):
    for j in range(n_clusters):
        ax.text(j, i, f"{M_cluster[i,j]:+.4f}", ha='center', va='center', fontsize=10)
plt.colorbar(im, ax=ax, label="ΔCE moyen inter-cluster")
ax.set_title("Interaction entre clusters — Super-language features",
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_cluster_interaction.png"), dpi=150)
plt.close()
print("→ 5_cluster_interaction.png")

# ── Plot 6 — Évolution par layer pour des paires choisies ────────────────────
if M_per_layer is not None:
    interesting_pairs = [
        ("fr", "en"), ("fr", "zh"), ("fr", "ja"),
        ("zh", "en"), ("es", "fr"), ("ko", "ja"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (lang_eval, lang_ablate) in zip(axes, interesting_pairs):
        if lang_eval not in LANGUAGES or lang_ablate not in LANGUAGES:
            ax.axis('off')
            continue
        i    = LANGUAGES.index(lang_eval)
        j    = LANGUAGES.index(lang_ablate)
        vals = M_per_layer[i, j, :]
        ax.plot(LAYERS, vals, marker='o', color='steelblue', linewidth=2, markersize=6)
        ax.axhline(float(M[i, j]), color='tomato', linestyle='--',
                   linewidth=1.2, label=f"Moy. = {M[i,j]:.3f}")
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_title(f"Ablate [{lang_ablate.upper()}] → eval [{lang_eval.upper()}]",
                     fontweight='bold', fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("ΔCE")
        ax.legend(fontsize=8)
        ax.grid(linestyle='--', alpha=0.4)
    fig.suptitle("Évolution du ΔCE par layer pour des paires choisies",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_delta_ce_per_layer.png"), dpi=150)
    plt.close()
    print("→ 6_delta_ce_per_layer.png")

# ── Plot 7 — Asymétrie ────────────────────────────────────────────────────────
asym_data = []
for i in range(n_langs):
    for j in range(i+1, n_langs):
        asym_data.append({
            "pair"     : f"{LANGUAGES[i].upper()}↔{LANGUAGES[j].upper()}",
            "M_ij"     : M[i, j],
            "M_ji"     : M[j, i],
            "asymmetry": abs(M[i, j] - M[j, i]),
        })
df_asym = pd.DataFrame(asym_data).sort_values("asymmetry", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
top15 = df_asym.head(15)
x = range(len(top15))
axes[0].bar(x, top15["M_ij"], width=0.4, label="M[i,j]", color='steelblue', alpha=0.8)
axes[0].bar([v+0.4 for v in x], top15["M_ji"], width=0.4,
            label="M[j,i]", color='tomato', alpha=0.8)
axes[0].set_xticks([v+0.2 for v in x])
axes[0].set_xticklabels(top15["pair"], rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel("ΔCE moyen")
axes[0].set_title("Top-15 paires les plus asymétriques", fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

max_val = max(df_asym["M_ij"].max(), df_asym["M_ji"].max()) * 1.1
axes[1].scatter(df_asym["M_ij"], df_asym["M_ji"], alpha=0.7, s=60, color='slateblue')
axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label="Symétrie parfaite")
for _, row in df_asym.head(5).iterrows():
    axes[1].annotate(row["pair"], (row["M_ij"], row["M_ji"]), fontsize=7)
axes[1].set_xlabel("M[i,j]")
axes[1].set_ylabel("M[j,i]")
axes[1].set_title("Symétrie de l'influence entre langues", fontweight='bold')
axes[1].legend()
axes[1].grid(linestyle='--', alpha=0.4)
plt.suptitle("Analyse d'asymétrie (moyenne sur toutes les layers)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_asymmetry_analysis.png"), dpi=150)
plt.close()
print("→ 7_asymmetry_analysis.png")

# ── CSV ───────────────────────────────────────────────────────────────────────
pd.DataFrame(M, index=LANGUAGES, columns=LANGUAGES).to_csv(
    os.path.join(OUTPUT_DIR, "interaction_matrix_avg_layers.csv")
)
print("→ interaction_matrix_avg_layers.csv")

print(f"\n✅ Tout sauvegardé dans '{OUTPUT_DIR}/'  —  7 plots + 1 CSV")