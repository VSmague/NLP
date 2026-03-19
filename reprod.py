"""
reproduce_figures_section5.py
==============================
Reproduit les Figures 5 et 6 de la section 5 de :
  "Unveiling Language-Specific Features in LLMs via Sparse Autoencoders"
  Deng et al., ACL 2025

Figure 5 : ΔCE après ablation du top-1 feature d'une langue cible,
           évalué sur TOUTES les langues.
           → Montre que l'ablation n'impacte significativement
             que la langue cible.

Figure 6 : ΔCE pour 3 langues (cible + 2 autres) en ablating
           top-1 seul, top-2 seul, top-1+2 ensemble des features FR.
           → Montre l'effet synergique entre features.
           Reproduit avec toutes les layers (courbe par layer).

Prérequis : avoir tourné ablation.py une fois
            (sae_features/ doit exister)
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

# ── Paramètres ────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-0.6B"
SAE_RELEASE = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR    = "sae_features_small"
OUTPUT_DIR  = "figures_section5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FIG5 = os.path.join(SAVE_DIR, "fig5_cache.pkl")
CACHE_FIG6 = os.path.join(SAVE_DIR, "fig6_cache.pkl")

# Langue dont on ablate les features (paper utilise FR pour Fig 6)
LANG_ABLATE = "fr"

# Pour Fig 6 : les 3 langues évaluées (cible + 2 contrôles)
# Paper : French (cible), + 2 autres — on prend Japanese et Chinese comme contrôles
LANGS_FIG6  = ["fr", "ja", "zh"]

N_SENTENCES = 100   # phrases par langue — mettre True pour tout le dataset
RECOMPUTE   = False  # True pour forcer le recalcul

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese',
    'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic',
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
    bar     = "█" * int(30 * done / total) + "░" * (30 - int(30 * done / total))
    print(f"\r{prefix} [{bar}] {pct:5.1f}%  elapsed={fmt_time(elapsed)}  ETA={fmt_time(eta)}",
          end="", flush=True)
    if done == total:
        print()


# =============================================================================
# Chargement du cache SAE + données
# =============================================================================

print("\n[0] Chargement du cache SAE...")
with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)

LAYERS    = meta["layers"]   # layers disponibles dans le cache
n_langs   = len(TARGET_LANGUAGES)

top_index_all = {}
for layer in LAYERS:
    top_index_all[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"), weights_only=True
    )
print(f"   {len(LAYERS)} layers chargées : {LAYERS}")


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
    """ΔCE moyen sur une liste de phrases pour une ablation donnée."""
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
# Chargement modèle + données (seulement si nécessaire)
# =============================================================================

need_compute = (RECOMPUTE or
                not os.path.exists(CACHE_FIG5) or
                not os.path.exists(CACHE_FIG6))

if need_compute:
    import requests

    print("\n[1] Chargement modèle...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    print(f"   Chargé en {fmt_time(time.time()-t0)}")

    print("\n[2] Chargement données...")
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
    print(f"   {len(multilingual_texts)} phrases chargées")


# =============================================================================
# FIGURE 5
# ─────────────────────────────────────────────────────────────────────────────
# Pour chaque langue L :
#   Ablater top-1 feature de L
#   Mesurer ΔCE sur TOUTES les langues
#
# Résultat : matrice (n_langs_ablate × n_langs_eval) × n_layers
# On moyenne sur les layers pour avoir une valeur par paire.
# =============================================================================

if RECOMPUTE or not os.path.exists(CACHE_FIG5):
    print("\n[Fig 5] Calcul ΔCE — top-1 ablation de chaque langue sur toutes les langues...")
    print(f"   {len(LAYERS)} layers × {len(TARGET_LANGUAGES)}² paires × {N_SENTENCES} phrases\n")

    # fig5_data[lang_ablate][lang_eval] = liste de ΔCE par layer
    fig5_data = {la: {le: [] for le in TARGET_LANGUAGES} for la in TARGET_LANGUAGES}

    t0    = time.time()
    total = len(TARGET_LANGUAGES) * len(LAYERS)
    done  = 0

    for lang_ablate in TARGET_LANGUAGES:
        j = TARGET_LANGUAGES.index(lang_ablate)

        for layer in LAYERS:
            sae      = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)
            feat_top1 = top_index_all[layer][j, :1].to(device)

            for lang_eval in TARGET_LANGUAGES:
                delta = compute_delta_ce(
                    model, sae, layer, feat_top1,
                    sentences_per_lang[lang_eval], tokenizer
                )
                fig5_data[lang_ablate][lang_eval].append(delta)

            del sae
            done += 1
            progress(done, total, t0, prefix="[Fig5]")

    # Moyenne sur les layers
    fig5_mean = {
        la: {le: float(np.mean(fig5_data[la][le])) for le in TARGET_LANGUAGES}
        for la in TARGET_LANGUAGES
    }

    with open(CACHE_FIG5, "wb") as f:
        pickle.dump({"fig5_data": fig5_data, "fig5_mean": fig5_mean}, f)
    print(f"\n   ✅ Fig5 sauvegardé — {fmt_time(time.time()-t0)}")

else:
    print("\n[Fig 5] Chargement depuis le cache...")
    with open(CACHE_FIG5, "rb") as f:
        cache5 = pickle.load(f)
    fig5_data = cache5["fig5_data"]
    fig5_mean = cache5["fig5_mean"]
    print("   ✅ Chargé")


# =============================================================================
# FIGURE 6
# ─────────────────────────────────────────────────────────────────────────────
# Pour LANG_ABLATE (French par défaut) :
#   Config A : ablate top-1 uniquement
#   Config B : ablate top-2 uniquement
#   Config C : ablate top-1 + top-2 ensemble
# Évalué sur LANGS_FIG6 (la cible + 2 contrôles), par layer.
# =============================================================================

if RECOMPUTE or not os.path.exists(CACHE_FIG6):
    print(f"\n[Fig 6] Calcul ΔCE synergique — features {LANG_ABLATE.upper()}...")
    print(f"   Configs : top-1 | top-2 | top-1+2")
    print(f"   Langues évaluées : {[l.upper() for l in LANGS_FIG6]}")
    print(f"   {len(LAYERS)} layers × 3 configs × {len(LANGS_FIG6)} langues × {N_SENTENCES} phrases\n")

    j = TARGET_LANGUAGES.index(LANG_ABLATE)

    # fig6_data[config][lang_eval][layer_idx] = ΔCE
    fig6_data = {
        "top-1"   : {le: [] for le in LANGS_FIG6},
        "top-2"   : {le: [] for le in LANGS_FIG6},
        "top-1+2" : {le: [] for le in LANGS_FIG6},
    }

    t0    = time.time()
    total = len(LAYERS)

    for li, layer in enumerate(LAYERS):
        sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)

        feat_top1  = top_index_all[layer][j, 0:1].to(device)   # top-1 seulement
        feat_top2  = top_index_all[layer][j, 1:2].to(device)   # top-2 seulement
        feat_both  = top_index_all[layer][j, 0:2].to(device)   # top-1 ET top-2

        for lang_eval in LANGS_FIG6:
            sents = sentences_per_lang[lang_eval]
            fig6_data["top-1"  ][lang_eval].append(
                compute_delta_ce(model, sae, layer, feat_top1, sents, tokenizer))
            fig6_data["top-2"  ][lang_eval].append(
                compute_delta_ce(model, sae, layer, feat_top2, sents, tokenizer))
            fig6_data["top-1+2"][lang_eval].append(
                compute_delta_ce(model, sae, layer, feat_both, sents, tokenizer))

        del sae
        progress(li + 1, total, t0, prefix="[Fig6]")

    with open(CACHE_FIG6, "wb") as f:
        pickle.dump({"fig6_data": fig6_data, "layers": LAYERS,
                     "lang_ablate": LANG_ABLATE, "langs_eval": LANGS_FIG6}, f)
    print(f"\n   ✅ Fig6 sauvegardé — {fmt_time(time.time()-t0)}")

else:
    print("\n[Fig 6] Chargement depuis le cache...")
    with open(CACHE_FIG6, "rb") as f:
        cache6 = pickle.load(f)
    fig6_data   = cache6["fig6_data"]
    LAYERS      = cache6["layers"]
    LANG_ABLATE = cache6["lang_ablate"]
    LANGS_FIG6  = cache6["langs_eval"]
    print("   ✅ Chargé")


# =============================================================================
# PLOT — Figure 5
# ─────────────────────────────────────────────────────────────────────────────
# Reproduit le style du papier :
# Pour chaque langue ablated : barplot ΔCE sur toutes les langues évaluées
# Langue cible en couleur distincte, autres en gris
# =============================================================================

print("\n[Plot] Génération Figure 5...")

# On plot les langues non-anglaises comme dans le papier (English excluded)
LANGS_PLOT = [l for l in TARGET_LANGUAGES if l != 'en']
n_plot     = len(LANGS_PLOT)

fig, axes = plt.subplots(
    3, 3, figsize=(14, 11),
    gridspec_kw={'hspace': 0.5, 'wspace': 0.35}
)
axes = axes.flatten()

COLORS_EVAL = {
    lang: '#D62728' if lang == lang else '#AAAAAA'
    for lang in TARGET_LANGUAGES
}
BAR_COLOR_TARGET  = '#D62728'   # rouge = langue cible
BAR_COLOR_OTHER   = '#AEC6CF'   # bleu pâle = autres langues
BAR_COLOR_EN      = '#888888'   # gris = anglais (toujours en dernier)

for ax_idx, lang_ablate in enumerate(LANGS_PLOT):
    ax = axes[ax_idx]

    # Ordonner : langue cible en premier, english en dernier, reste trié par ΔCE
    others = sorted(
        [l for l in TARGET_LANGUAGES if l != lang_ablate and l != 'en'],
        key=lambda l: -fig5_mean[lang_ablate][l]
    )
    order  = [lang_ablate] + others + ['en']
    values = [fig5_mean[lang_ablate][le] for le in order]
    colors = (
        [BAR_COLOR_TARGET] +
        [BAR_COLOR_OTHER] * len(others) +
        [BAR_COLOR_EN]
    )
    xlabels = [LANG_NAMES[l] for l in order]

    bars = ax.bar(range(len(order)), values, color=colors,
                  width=0.7, edgecolor='white', linewidth=0.5)

    # Valeur au dessus de chaque barre
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.2f}", ha='center', va='bottom',
                fontsize=6.5, color='#333333')

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=7.5)
    ax.set_title(f"Ablate {LANG_NAMES[lang_ablate]} feature",
                 fontsize=9, fontweight='bold', pad=4)
    ax.set_ylabel("ΔCE", fontsize=8)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annoter la langue cible
    ax.get_xticklabels()[0].set_color(BAR_COLOR_TARGET)
    ax.get_xticklabels()[0].set_fontweight('bold')

# Titre global + légende
fig.suptitle(
    "Figure 5 — ΔCE après ablation du top-1 feature de chaque langue\n"
    "L'ablation impacte significativement la langue cible (rouge) "
    "mais peu les autres",
    fontsize=11, fontweight='bold', y=1.01
)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=BAR_COLOR_TARGET, label='Langue cible (ablated)'),
    Patch(facecolor=BAR_COLOR_OTHER,  label='Autres langues'),
    Patch(facecolor=BAR_COLOR_EN,     label='English'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=9, framealpha=0.5, bbox_to_anchor=(0.5, -0.03))

plt.savefig(os.path.join(OUTPUT_DIR, "figure5_ablation_specificity.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("   → figure5_ablation_specificity.png")


# =============================================================================
# PLOT — Figure 6
# ─────────────────────────────────────────────────────────────────────────────
# Reproduit le style du papier :
# 3 sous-plots (un par langue évaluée), courbe ΔCE par layer
# 3 configs : top-1, top-2, top-1+2
# =============================================================================

print("[Plot] Génération Figure 6...")

CONFIG_COLORS = {
    "top-1"   : "#4C72B0",   # bleu
    "top-2"   : "#DD8452",   # orange
    "top-1+2" : "#C44E52",   # rouge (synergie)
}
CONFIG_LABELS = {
    "top-1"   : "Ablate top-1 only",
    "top-2"   : "Ablate top-2 only",
    "top-1+2" : "Ablate top-1 & top-2 (synergy)",
}
CONFIG_STYLES = {
    "top-1"   : "-",
    "top-2"   : "--",
    "top-1+2" : "-",
}
CONFIG_WIDTHS = {
    "top-1"   : 1.8,
    "top-2"   : 1.8,
    "top-1+2" : 2.5,
}

fig, axes = plt.subplots(1, len(LANGS_FIG6), figsize=(5 * len(LANGS_FIG6), 5),
                          sharey=False)
if len(LANGS_FIG6) == 1:
    axes = [axes]

for ax, lang_eval in zip(axes, LANGS_FIG6):
    is_target = (lang_eval == LANG_ABLATE)

    for config in ["top-1", "top-2", "top-1+2"]:
        vals = fig6_data[config][lang_eval]
        ax.plot(LAYERS, vals,
                color=CONFIG_COLORS[config],
                linestyle=CONFIG_STYLES[config],
                linewidth=CONFIG_WIDTHS[config],
                marker='o', markersize=4,
                label=CONFIG_LABELS[config])

    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("ΔCE loss", fontsize=10)

    title_color = '#D62728' if is_target else '#333333'
    title_suffix = " ← TARGET" if is_target else ""
    ax.set_title(
        f"Eval on {LANG_NAMES[lang_eval]} text{title_suffix}",
        fontsize=10, fontweight='bold', color=title_color
    )

    ax.grid(linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annoter la synergy au meilleur layer
    best_layer_idx = int(np.argmax(fig6_data["top-1+2"][lang_eval]))
    best_val       = fig6_data["top-1+2"][lang_eval][best_layer_idx]
    top1_val       = fig6_data["top-1"][lang_eval][best_layer_idx]
    top2_val       = fig6_data["top-2"][lang_eval][best_layer_idx]
    synergy        = best_val - (top1_val + top2_val)

    if is_target and synergy > 0.01:
        ax.annotate(
            f"synergy\n+{synergy:.3f}",
            xy=(LAYERS[best_layer_idx], best_val),
            xytext=(LAYERS[best_layer_idx] + 1, best_val + 0.05),
            fontsize=7.5, color=CONFIG_COLORS["top-1+2"],
            arrowprops=dict(arrowstyle='->', color=CONFIG_COLORS["top-1+2"],
                            lw=1.2),
        )

# Légende commune
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3,
           fontsize=9, framealpha=0.5, bbox_to_anchor=(0.5, -0.08))

fig.suptitle(
    f"Figure 6 — ΔCE après ablation des features {LANG_NAMES[LANG_ABLATE]}\n"
    f"L'ablation simultanée top-1+2 produit un effet synergique sur le {LANG_NAMES[LANG_ABLATE]} "
    f"mais pas sur les autres langues",
    fontsize=11, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.08, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "figure6_synergy.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("   → figure6_synergy.png")


# =============================================================================
# PLOT BONUS — Figure 6 style "par langue" comme dans le papier
# Pour chaque langue de LANGS_FIG6 : 3 barres (top1, top2, top1+2)
# au meilleur layer, pour voir clairement la synergie
# =============================================================================

print("[Plot] Génération Figure 6 — version résumé par langue...")

fig, ax = plt.subplots(figsize=(9, 5))

n_langs_eval = len(LANGS_FIG6)
n_configs    = 3
width        = 0.22
configs_list = ["top-1", "top-2", "top-1+2"]

for ci, config in enumerate(configs_list):
    # Utiliser le max sur toutes les layers comme valeur représentative
    vals = [max(fig6_data[config][le]) for le in LANGS_FIG6]
    x    = np.arange(n_langs_eval) + ci * width
    bars = ax.bar(x, vals, width=width,
                  color=CONFIG_COLORS[config],
                  label=CONFIG_LABELS[config],
                  alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8)

ax.set_xticks(np.arange(n_langs_eval) + width)
ax.set_xticklabels([LANG_NAMES[l] for l in LANGS_FIG6], fontsize=11)
ax.set_ylabel("Max ΔCE (sur toutes les layers)", fontsize=10)
ax.set_title(
    f"Figure 6 (résumé) — Effet synergique des features {LANG_NAMES[LANG_ABLATE]}\n"
    "top-1+2 > top-1 + top-2 sur la langue cible, effet nul ailleurs",
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Flèche synergie sur la langue cible
target_idx    = LANGS_FIG6.index(LANG_ABLATE)
val_top1      = max(fig6_data["top-1"][LANG_ABLATE])
val_top2      = max(fig6_data["top-2"][LANG_ABLATE])
val_both      = max(fig6_data["top-1+2"][LANG_ABLATE])
synergy_val   = val_both - (val_top1 + val_top2)
if synergy_val > 0:
    ax.annotate(
        f"Synergie\n+{synergy_val:.3f}",
        xy=(target_idx + 2 * width, val_both),
        xytext=(target_idx + 2 * width + 0.5, val_both + 0.05),
        fontsize=9, color=CONFIG_COLORS["top-1+2"], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=CONFIG_COLORS["top-1+2"], lw=1.5),
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure6_synergy_summary.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("   → figure6_synergy_summary.png")

print(f"\n✅ Toutes les figures sauvegardées dans '{OUTPUT_DIR}/'")
print("   figure5_ablation_specificity.png")
print("   figure6_synergy.png")
print("   figure6_synergy_summary.png")