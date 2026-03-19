"""
cross_language_synergy.py
==========================
Teste si ablater les top features de deux langues simultanément
produit un effet synergique, selon qu'elles sont dans le même
cluster ou non.

Configs testées pour une paire (A, B) évaluée sur texte de langue A :
  - Baseline
  - Ablate top-1 de A seul
  - Ablate top-1 de B seul
  - Ablate top-1 A + top-1 B ensemble

Synergie = (A+B ensemble) - (A seul + B seul)
  → positive si les features interagissent
  → ~0 si les features sont indépendantes

Comparaison :
  - Paires intra-cluster  (A et B dans le même cluster)  → synergie attendue
  - Paires inter-cluster  (A et B dans des clusters diff) → pas de synergie

Prérequis : sae_features/ doit exister (ablation.py tourné une fois)
            interaction_matrix.csv doit exister
"""

import os
import json
import pickle
import configparser
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
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
OUTPUT_DIR  = "plots_synergy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FILE  = os.path.join(SAVE_DIR, "cross_language_synergy.pkl")
RECOMPUTE   = False

N_CLUSTERS  = 3      # nombre de clusters souhaité
N_SENTENCES = 50     # phrases par langue pour estimer le delta-CE
USE_BEST_LAYER = True  # True = meilleure layer par langue, False = FIXED_LAYER
FIXED_LAYER    = 18

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']

TYPO_FAMILIES = {
    'en': 'Indo-European (Germanic)', 'es': 'Indo-European (Romance)',
    'fr': 'Indo-European (Romance)',  'pt': 'Indo-European (Romance)',
    'ar': 'Afro-Asiatic (Semitic)',   'zh': 'Sino-Tibetan',
    'ja': 'Japonic',                  'ko': 'Koreanic',
    'vi': 'Austroasiatic',            'th': 'Kra-Dai',
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


# ── Utilitaire temps ──────────────────────────────────────────────────────────

def fmt_time(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

def progress(done, total, t0, prefix=""):
    elapsed = time.time() - t0
    eta     = (elapsed / done) * (total - done) if done > 0 else 0
    pct     = done / total * 100
    bar     = "█" * int(30*done/total) + "░" * (30-int(30*done/total))
    print(f"\r{prefix} [{bar}] {pct:5.1f}%  elapsed={fmt_time(elapsed)}  ETA={fmt_time(eta)}",
          end="", flush=True)
    if done == total:
        print()


# =============================================================================
# 1. Charger cache SAE
# =============================================================================

print("\n[1] Chargement cache SAE...")
with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)

LAYERS  = meta["layers"]
n_langs = len(TARGET_LANGUAGES)

top_index_all  = {}
top_values_all = {}
for layer in LAYERS:
    top_index_all[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"), weights_only=True)
    top_values_all[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_values.pt"), weights_only=True)

# Meilleure layer par langue (v-score top-1 max)
best_layer_per_lang = {}
for i, lang in enumerate(TARGET_LANGUAGES):
    best_layer_per_lang[lang] = max(
        LAYERS, key=lambda l: top_values_all[l][i, 0].item()
    )
print(f"   Meilleure layer par langue :")
for lang, l in best_layer_per_lang.items():
    v = top_values_all[l][TARGET_LANGUAGES.index(lang), 0].item()
    print(f"     {lang.upper()} → layer {l}  (v-score={v:.4f})")


# =============================================================================
# 2. Construire les clusters depuis la matrice d'interaction
# =============================================================================

print("\n[2] Construction des clusters...")
M_df = pd.read_csv("plots_interaction/interaction_matrix.csv", index_col=0)
M    = M_df.values

# Distance symétrisée
D = np.zeros((n_langs, n_langs))
for i in range(n_langs):
    for j in range(n_langs):
        num   = M[i, j] + M[j, i]
        denom = M[i, i] + M[j, j]
        D[i, j] = 1.0 - (num / denom) if denom > 1e-8 else 1.0
np.fill_diagonal(D, 0)
D = np.clip((D + D.T) / 2, 0, None)

Z = linkage(squareform(D), method='ward')

# ── Correction 1 : N_CLUSTERS borné + cut_height robuste ─────────────────────
N_CLUSTERS = max(2, min(N_CLUSTERS, n_langs - 1))
cut_h      = 0.5 * (Z[-(N_CLUSTERS-1), 2] + Z[-N_CLUSTERS, 2])
clusters   = fcluster(Z, t=cut_h, criterion='distance')

lang_to_cluster = {TARGET_LANGUAGES[i]: int(clusters[i]) for i in range(n_langs)}

print(f"   {N_CLUSTERS} clusters demandés → {len(set(clusters))} obtenus :")
for c in sorted(set(clusters)):
    langs_in = [l for l in TARGET_LANGUAGES if lang_to_cluster[l] == c]
    singleton = " (singleton)" if len(langs_in) == 1 else ""
    print(f"   Cluster {c} : {[l.upper() for l in langs_in]}{singleton}")

# Paires intra et inter
all_pairs   = list(itertools.combinations(TARGET_LANGUAGES, 2))
intra_pairs = [(a, b) for a, b in all_pairs
               if lang_to_cluster[a] == lang_to_cluster[b]]
inter_pairs = [(a, b) for a, b in all_pairs
               if lang_to_cluster[a] != lang_to_cluster[b]]

print(f"\n   {len(intra_pairs)} paires intra  |  {len(inter_pairs)} paires inter")
if not intra_pairs:
    print("   ⚠️  Aucune paire intra-cluster — tous les clusters sont singletons.")
    print("      Augmente N_CLUSTERS ou vérifie ta matrice.")


# =============================================================================
# 3. Fonctions utilitaires
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


def ablate_two_layers(model, sae_a, layer_a, feat_a,
                              sae_b, layer_b, feat_b, inputs):
    """Ablate feat_a à layer_a ET feat_b à layer_b en un seul forward pass."""

    def make_hook(sae, feat_idx):
        W_dec     = sae.W_dec.T.to(device)
        dirs      = W_dec[:, feat_idx]
        norms     = torch.norm(dirs, dim=0) ** 2
        dirs_norm = dirs / norms.clamp(min=1e-8)
        def _hook(module, inp, output):
            act = output[0].to(torch.float32) if isinstance(output, tuple) \
                  else output.to(torch.float32)
            act_ablated = act - (act @ dirs_norm) @ dirs_norm.T
            if isinstance(output, tuple):
                return (act_ablated.to(output[0].dtype),) + output[1:]
            return act_ablated.to(output.dtype)
        return _hook

    if layer_a != layer_b:
        handles = [
            model.model.layers[layer_a].register_forward_hook(make_hook(sae_a, feat_a)),
            model.model.layers[layer_b].register_forward_hook(make_hook(sae_b, feat_b)),
        ]
    else:
        # Même layer : ablater les deux features ensemble en un seul hook
        W_dec    = sae_a.W_dec.T.to(device)
        both_idx = torch.cat([feat_a, feat_b])
        dirs     = W_dec[:, both_idx]
        norms    = torch.norm(dirs, dim=0) ** 2
        dirs_n   = dirs / norms.clamp(min=1e-8)
        def _hook_both(module, inp, output):
            act = output[0].to(torch.float32) if isinstance(output, tuple) \
                  else output.to(torch.float32)
            act_ablated = act - (act @ dirs_n) @ dirs_n.T
            if isinstance(output, tuple):
                return (act_ablated.to(output[0].dtype),) + output[1:]
            return act_ablated.to(output.dtype)
        handles = [model.model.layers[layer_a].register_forward_hook(_hook_both)]

    try:
        out = model.forward(inputs)
    finally:
        for h in handles:
            h.remove()
    return out


def compute_ce_loss(logits, inputs):
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = inputs[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels)


def run_pair_experiment(model, sae_a, layer_a, feat_a,
                                sae_b, layer_b, feat_b,
                                sentences, tokenizer):
    """
    Pour une paire (A, B) évaluée sur texte de langue A :
    Retourne dict avec les 4 configs et la synergie calculée.
    """
    losses = {"baseline": [], "only_A": [], "only_B": [], "A_and_B": []}

    for text in sentences:
        inp = tokenizer.encode(text, return_tensors='pt',
                               add_special_tokens=True).to(device)
        if inp.size(1) < 2:
            continue
        losses["baseline"].append(
            compute_ce_loss(model.forward(inp)["logits"], inp).item())
        losses["only_A"].append(
            compute_ce_loss(directional_ablation_forward(
                model, sae_a, layer_a, feat_a, inp)["logits"], inp).item())
        losses["only_B"].append(
            compute_ce_loss(directional_ablation_forward(
                model, sae_b, layer_b, feat_b, inp)["logits"], inp).item())
        losses["A_and_B"].append(
            compute_ce_loss(ablate_two_layers(
                model, sae_a, layer_a, feat_a,
                       sae_b, layer_b, feat_b, inp)["logits"], inp).item())

    base     = np.nanmean(losses["baseline"])
    delta_A  = np.nanmean(losses["only_A"])  - base
    delta_B  = np.nanmean(losses["only_B"])  - base
    delta_AB = np.nanmean(losses["A_and_B"]) - base
    synergy  = delta_AB - (delta_A + delta_B)

    return {
        "delta_A"  : float(delta_A),
        "delta_B"  : float(delta_B),
        "delta_AB" : float(delta_AB),
        "synergy"  : float(synergy),
        "additive" : float(delta_A + delta_B),
    }


# =============================================================================
# 4. Calcul — toutes les paires
# =============================================================================

if RECOMPUTE or not os.path.exists(CACHE_FILE):
    import requests

    print("\n[3] Chargement modèle...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    print(f"   Chargé en {fmt_time(time.time()-t0)}")

    print("[4] Chargement données...")
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

    print(f"\n[5] Calcul synergie — {len(all_pairs)} paires...")

    results  = {}
    t0       = time.time()
    sae_cache = {}

    def get_sae(layer):
        if layer not in sae_cache:
            if len(sae_cache) >= 4:
                old = next(iter(sae_cache))
                del sae_cache[old]
            sae_cache[layer] = SAE.from_pretrained(
                SAE_RELEASE, f"layer_{layer}").to(device)
        return sae_cache[layer]

    for done, (lang_a, lang_b) in enumerate(all_pairs, start=1):
        i_a = TARGET_LANGUAGES.index(lang_a)
        i_b = TARGET_LANGUAGES.index(lang_b)

        layer_a = best_layer_per_lang[lang_a] if USE_BEST_LAYER else FIXED_LAYER
        layer_b = best_layer_per_lang[lang_b] if USE_BEST_LAYER else FIXED_LAYER

        feat_a  = top_index_all[layer_a][i_a, 0:1].to(device)
        feat_b  = top_index_all[layer_b][i_b, 0:1].to(device)

        sae_a   = get_sae(layer_a)
        sae_b   = get_sae(layer_b)

        res = run_pair_experiment(
            model, sae_a, layer_a, feat_a,
                   sae_b, layer_b, feat_b,
            sentences_per_lang[lang_a], tokenizer
        )
        results[(lang_a, lang_b)] = res

        progress(done, len(all_pairs), t0, prefix="[Synergy]")
        c_type = "intra" if lang_to_cluster[lang_a] == lang_to_cluster[lang_b] else "inter"
        print(f"   {lang_a.upper()}+{lang_b.upper()} [{c_type}]  "
              f"ΔA={res['delta_A']:+.3f}  ΔB={res['delta_B']:+.3f}  "
              f"ΔAB={res['delta_AB']:+.3f}  syn={res['synergy']:+.3f}", flush=True)

    for sae in sae_cache.values():
        del sae

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "results"         : results,
            "lang_to_cluster" : lang_to_cluster,
            "all_pairs"       : all_pairs,
            "intra_pairs"     : intra_pairs,
            "inter_pairs"     : inter_pairs,
            "n_clusters"      : N_CLUSTERS,
        }, f)
    print(f"\n✅ Résultats sauvegardés dans {CACHE_FILE}")

else:
    print(f"\n[3-5] Chargement depuis le cache ({CACHE_FILE})...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    results         = cache["results"]
    lang_to_cluster = cache["lang_to_cluster"]
    all_pairs       = cache["all_pairs"]
    intra_pairs     = cache["intra_pairs"]
    inter_pairs     = cache["inter_pairs"]
    print("   ✅ Chargé")


# =============================================================================
# 5. Analyse et plots
# =============================================================================

syn_intra = [results[(a,b)]["synergy"] for a,b in intra_pairs if (a,b) in results]
syn_inter = [results[(a,b)]["synergy"] for a,b in inter_pairs if (a,b) in results]

print(f"\n── Résumé synergies ──")
if syn_intra:
    print(f"  Intra-cluster : mean={np.mean(syn_intra):+.4f}  std={np.std(syn_intra):.4f}  n={len(syn_intra)}")
else:
    print("  Intra-cluster : aucune paire (tous singletons)")
if syn_inter:
    print(f"  Inter-cluster : mean={np.mean(syn_inter):+.4f}  std={np.std(syn_inter):.4f}  n={len(syn_inter)}")


# ── Plot 1 — Distribution des synergies intra vs inter ────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if syn_intra:
    axes[0].hist(syn_intra, bins=max(4, len(syn_intra)//2),
                 alpha=0.7, color='#C44E52', label=f'Intra-cluster (n={len(syn_intra)})')
if syn_inter:
    axes[0].hist(syn_inter, bins=max(4, len(syn_inter)//2),
                 alpha=0.7, color='#4C72B0', label=f'Inter-cluster (n={len(syn_inter)})')

axes[0].axvline(0, color='grey', linestyle='--', linewidth=1, label='Zéro')

# ── Correction 2 : axvline seulement si la liste n'est pas vide ──────────────
if syn_intra:
    axes[0].axvline(np.mean(syn_intra), color='#C44E52', linestyle='-', linewidth=2,
                    label=f'Moy. intra = {np.mean(syn_intra):+.3f}')
if syn_inter:
    axes[0].axvline(np.mean(syn_inter), color='#4C72B0', linestyle='-', linewidth=2,
                    label=f'Moy. inter = {np.mean(syn_inter):+.3f}')

axes[0].set_xlabel("Synergie (ΔAB − ΔA − ΔB)", fontsize=11)
axes[0].set_ylabel("Nombre de paires")
axes[0].set_title("Distribution des synergies\nintra vs inter-cluster", fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Boxplot
data_bp   = [x for x in [syn_intra, syn_inter] if x]
labels_bp = [l for l, x in [('Intra-cluster', syn_intra), ('Inter-cluster', syn_inter)] if x]
colors_bp = [c for c, x in [('#C44E52', syn_intra), ('#4C72B0', syn_inter)] if x]

if data_bp:
    bp = axes[1].boxplot(
        data_bp, labels=labels_bp, patch_artist=True,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=5, alpha=0.6),
    )
    for box, color in zip(bp['boxes'], colors_bp):
        box.set_facecolor(color)
        box.set_alpha(0.7)
else:
    axes[1].text(0.5, 0.5, "Pas assez de données", ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=12, color='grey')

axes[1].axhline(0, color='grey', linestyle='--', linewidth=1)
axes[1].set_ylabel("Synergie (ΔAB − ΔA − ΔB)", fontsize=11)
axes[1].set_title("Boxplot des synergies\nintra vs inter-cluster", fontweight='bold')
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

fig.suptitle("Synergie cross-linguistique : les langues du même cluster\n"
             "s'influencent-elles davantage ?",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_synergy_distribution.png"), dpi=150)
plt.close()
print("→ 1_synergy_distribution.png")


# ── Plot 2 — Barplot par paire ────────────────────────────────────────────────

all_pairs_sorted = sorted(
    all_pairs,
    key=lambda p: (lang_to_cluster[p[0]] == lang_to_cluster[p[1]],
                   results.get(p, {}).get("synergy", 0)),
    reverse=True
)

labels_pairs = [f"{a.upper()}\n+{b.upper()}" for a, b in all_pairs_sorted]
syn_vals     = [results[(a,b)]["synergy"] for a,b in all_pairs_sorted]
colors_pairs = [
    '#C44E52' if lang_to_cluster[a] == lang_to_cluster[b] else '#4C72B0'
    for a, b in all_pairs_sorted
]

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(range(len(all_pairs_sorted)), syn_vals, color=colors_pairs, alpha=0.85, width=0.7)
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
if syn_intra:
    ax.axhline(np.mean(syn_intra), color='#C44E52', linestyle=':', linewidth=1.5,
               label=f'Moy. intra = {np.mean(syn_intra):+.3f}')
if syn_inter:
    ax.axhline(np.mean(syn_inter), color='#4C72B0', linestyle=':', linewidth=1.5,
               label=f'Moy. inter = {np.mean(syn_inter):+.3f}')
ax.set_xticks(range(len(all_pairs_sorted)))
ax.set_xticklabels(labels_pairs, fontsize=8)
ax.set_ylabel("Synergie (ΔAB − ΔA − ΔB)")
ax.set_title("Synergie par paire (rouge = intra-cluster, bleu = inter-cluster)",
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_synergy_per_pair.png"), dpi=150)
plt.close()
print("→ 2_synergy_per_pair.png")


# ── Plot 3 — Style Figure 6 du papier ────────────────────────────────────────

# ── Correction 3 : n_show adapté à ce qui est disponible ─────────────────────
n_show    = min(3, len(intra_pairs), len(inter_pairs))

if n_show == 0:
    print("   ⚠️  Plot 3 ignoré : pas assez de paires intra ET inter.")
else:
    top_intra = sorted(intra_pairs, key=lambda p: results[p]["synergy"],
                       reverse=True)[:n_show]
    top_inter = sorted(inter_pairs, key=lambda p: results[p]["synergy"])[:n_show]
    selected  = top_intra + top_inter

    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 9), squeeze=False)

    for row, pairs_row in enumerate([top_intra, top_inter]):
        for col, (lang_a, lang_b) in enumerate(pairs_row):
            ax       = axes[row][col]
            res      = results[(lang_a, lang_b)]
            is_intra = lang_to_cluster[lang_a] == lang_to_cluster[lang_b]
            c_type   = "intra-cluster" if is_intra else "inter-cluster"
            color    = '#C44E52' if is_intra else '#4C72B0'

            configs    = ["ΔA seul", "ΔB seul", "ΔAB additif\n(prédit)", "ΔAB réel"]
            values     = [res["delta_A"], res["delta_B"], res["additive"], res["delta_AB"]]
            bar_colors = ['#888888', '#AAAAAA', '#DDDDDD', color]

            ax.bar(range(4), values, color=bar_colors, alpha=0.85, width=0.6)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.7)

            # Flèche synergie
            if abs(res["synergy"]) > 0.005:
                ax.annotate("",
                    xy=(3, res["delta_AB"]), xytext=(3, res["additive"]),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))
                ax.text(3.35, (res["additive"] + res["delta_AB"]) / 2,
                        f"syn={res['synergy']:+.3f}",
                        fontsize=8, color=color, va='center')

            ax.set_xticks(range(4))
            ax.set_xticklabels(configs, fontsize=8.5)
            ax.set_ylabel("ΔCE", fontsize=9)
            ax.set_title(f"{lang_a.upper()} + {lang_b.upper()} [{c_type}]\n"
                         f"eval sur texte {lang_a.upper()}",
                         fontsize=9.5, fontweight='bold', color=color)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.suptitle("Style Figure 6 — synergie cross-linguistique\n"
                 "Ligne 1 : meilleures paires intra  |  Ligne 2 : paires inter",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_synergy_figure6_style.png"), dpi=150)
    plt.close()
    print("→ 3_synergy_figure6_style.png")


# ── Plot 4 — Matrice de synergie ──────────────────────────────────────────────

syn_matrix = np.zeros((n_langs, n_langs))
for (a, b), res in results.items():
    i = TARGET_LANGUAGES.index(a)
    j = TARGET_LANGUAGES.index(b)
    syn_matrix[i, j] = res["synergy"]
    syn_matrix[j, i] = res["synergy"]

lang_labels = [l.upper() for l in TARGET_LANGUAGES]
vmax        = max(abs(syn_matrix).max(), 1e-6)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(syn_matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(n_langs))
ax.set_yticks(range(n_langs))
ax.set_xticklabels(lang_labels, fontsize=11)
ax.set_yticklabels(lang_labels, fontsize=11)

for i in range(n_langs):
    for j in range(n_langs):
        if i != j:
            is_intra = lang_to_cluster[TARGET_LANGUAGES[i]] == \
                       lang_to_cluster[TARGET_LANGUAGES[j]]
            ax.text(j, i, f"{syn_matrix[i,j]:+.3f}",
                    ha='center', va='center', fontsize=7.5,
                    fontweight='bold' if is_intra else 'normal',
                    color='black' if abs(syn_matrix[i,j]) < vmax*0.6 else 'white')

# ── Correction 4 : encadrement robuste — fonctionne aussi pour les singletons ─
for c in sorted(set(lang_to_cluster.values())):
    idxs = [i for i, l in enumerate(TARGET_LANGUAGES) if lang_to_cluster[l] == c]
    x0   = min(idxs) - 0.5
    x1   = max(idxs) + 0.5
    size = x1 - x0
    ax.add_patch(plt.Rectangle(
        (x0, x0), size, size,
        fill=False,
        edgecolor='black' if len(idxs) > 1 else 'grey',
        linewidth=2.5    if len(idxs) > 1 else 1.0,
        linestyle='-'    if len(idxs) > 1 else '--',
    ))

plt.colorbar(im, ax=ax, label="Synergie (ΔAB − ΔA − ΔB)")
ax.set_title("Matrice de synergie cross-linguistique\n"
             "(encadrés pleins = clusters multi-langues  |  pointillés = singletons)",
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_synergy_matrix.png"), dpi=150)
plt.close()
print("→ 4_synergy_matrix.png")

print(f"\n✅ Plots sauvegardés dans '{OUTPUT_DIR}/'")