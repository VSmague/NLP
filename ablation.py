"""
Language-Specific Features via SAE — Ablation Study
Reproducing the ablation from:
  "Unveiling Language-Specific Features in LLMs via Sparse Autoencoders" (Deng et al.)

Model : Qwen3-0.6B
SAE   : mwhanna-qwen3-0.6b-transcoders-lowl0
"""

import os
import configparser

# pip install sae-lens transformers

import json
import requests
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE


config = configparser.ConfigParser()
config.read("secrets.ini")

os.environ["HF_TOKEN"] = config["huggingface"]["token"]

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Parameters — edit these ───────────────────────────────────────────────────


SAVE_DIR = "sae_features"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_ID        = "Qwen/Qwen3-0.6B"   # or local path e.g. "qwen06b_local"
SAE_RELEASE     = "mwhanna-qwen3-0.6b-transcoders-lowl0"
ABLATION_LAYERS = range(28)
N_TEXTS_PER_LAN = 100    # texts used to compute v scores (use 50+ for final results)
LANG_TO_ABLATE  = "fr"  # language whose top features we ablate
OTHER_LANGUAGE  = "ja"  # control language (should be unaffected)
N_SENTENCES     = 100    # sentences used to measure delta-CE (use 50+ for final results)

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']

# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# 1. Load model & tokenizer
# =============================================================================

print("\n[1] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.eval()
print(f"    Loaded — {len(model.model.layers)} layers")


# =============================================================================
# 2. Load multilingual dataset
# =============================================================================

print("\n[2] Loading multilingual data...")
DATA_URL = "https://raw.githubusercontent.com/Aatrox103/multilingual-llm-features/main/SAE/data/multilingual_data.jsonl"
response = requests.get(DATA_URL)
response.raise_for_status()

multilingual_texts = []
for line in response.text.strip().split('\n'):
    obj = json.loads(line)
    multilingual_texts.append(obj['text'] if isinstance(obj, dict) else obj)

print(f"    Loaded {len(multilingual_texts)} sentences")


# =============================================================================
# 3. Core utilities
# =============================================================================

def gather_residual_activations(model, target_layer, inputs):
    """Return residual-stream output of target_layer. Shape: (1, seq, d_model)."""
    target_act = None

    def _hook(mod, inp, out):
        nonlocal target_act
        target_act = out[0]
        return out

    handle = model.model.layers[target_layer].register_forward_hook(_hook)
    model.forward(inputs)
    handle.remove()
    return target_act


def compute_top_index_per_lan_for_layer(
    model, sae, layer, target_lan, multilingual_texts, device, n_texts_per_lan=10
):
    """
    Compute language-specificity score v for each feature at a given layer.

    v_i(f) = mean_act_lang_i(f) - mean_{j!=i} mean_act_lang_j(f)

    Returns
    -------
    top_index_per_lan  : (num_langs, d_sae)  feature indices sorted by v desc
    top_values_per_lan : (num_langs, d_sae)  corresponding v values
    """
    sae_acts_per_lan = {}

    for i, lan in enumerate(target_lan):
        lang_texts = multilingual_texts[i * 100 : i * 100 + n_texts_per_lan]
        activations = []
        for text in lang_texts:
            inp = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True).to(device)
            act = gather_residual_activations(model, layer, inp)   # (1, seq, d_model)
            sae_act = sae.encode(act).cpu()                        # (1, seq, d_sae)
            if sae_act.dim() == 2:
                sae_act = sae_act.unsqueeze(0)
            activations.append(sae_act)
        sae_acts_per_lan[lan] = activations

    # average over (sentences x tokens) -> (num_langs, d_sae)
    avg_act_per_lan = []
    for lan in target_lan:
        stacked = torch.cat(sae_acts_per_lan[lan], dim=1)      # (1, total_tokens, d_sae)
        avg_act_per_lan.append(stacked.mean(dim=1).squeeze(0)) # (d_sae,)
    avg_act_per_lan = torch.stack(avg_act_per_lan)             # (num_langs, d_sae)

    top_index_per_lan, top_values_per_lan = [], []
    for i in range(len(target_lan)):
        mean_i = avg_act_per_lan[i]
        gamma  = torch.cat([avg_act_per_lan[:i], avg_act_per_lan[i+1:]]).mean(0)
        v = mean_i - gamma
        sorted_vals, sorted_idx = torch.sort(v, descending=True)
        top_index_per_lan.append(sorted_idx)
        top_values_per_lan.append(sorted_vals)

    return torch.stack(top_index_per_lan), torch.stack(top_values_per_lan)


def directional_ablation_forward(model, sae, layer, feature_indices, inputs):
    W_dec     = sae.W_dec.T.to(device)
    dirs      = W_dec[:, feature_indices]
    norms     = torch.norm(dirs, dim=0) ** 2
    dirs_norm = dirs / norms.clamp(min=1e-8)

    def _hook(module, inp, output):
        # output peut être un tuple ou un objet custom — on extrait le tensor
        if isinstance(output, tuple):
            act = output[0].to(torch.float32)
        else:
            act = output.to(torch.float32)

        coeff = act @ dirs_norm
        act_ablated = act - coeff @ dirs_norm.T

        # On reconstruit la sortie dans le bon format
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
    """Standard LM next-token CE loss with input/label shift."""
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = inputs[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels)


def get_top_k_indices(top_index_per_lan, lang_idx, k, n_langs):
    if top_index_per_lan.dim() == 2:
        return top_index_per_lan[lang_idx, :k]
    n_f = top_index_per_lan.shape[0] // n_langs
    return top_index_per_lan[lang_idx * n_f : lang_idx * n_f + k]


def run_ablation(model, sae, layer, top_index_per_lan_layer,
                 lang_idx, sentences, tokenizer, device, n_langs):
    """
    Run 4 ablation configs on a list of sentences.

    Returns dict: config_name -> mean delta-CE (ablated - baseline)
    """
    top1   = get_top_k_indices(top_index_per_lan_layer, lang_idx, 1, n_langs).to(device)
    top2   = get_top_k_indices(top_index_per_lan_layer, lang_idx, 2, n_langs).to(device)[1:2]
    top1_2 = get_top_k_indices(top_index_per_lan_layer, lang_idx, 2, n_langs).to(device)

    configs = {
        "Baseline"         : None,
        "Ablate top-1"     : top1,
        "Ablate top-2"     : top2,
        "Ablate top-1 & 2" : top1_2,
    }
    losses = {name: [] for name in configs}

    for text in sentences:
        inp = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True).to(device)
        if inp.size(1) < 2:
            continue
        for name, feat_idx in configs.items():
            if feat_idx is None:
                out = model.forward(inp)
            else:
                out = directional_ablation_forward(model, sae, layer, feat_idx, inp)
            losses[name].append(compute_ce_loss(out["logits"], inp).item())

    baseline_mean = np.nanmean(losses["Baseline"])
    return {name: np.nanmean(vals) - baseline_mean for name, vals in losses.items()}


# =============================================================================
# SAUVEGARDER
# =============================================================================

def save_feature_rankings(top_index_per_layer, top_values_per_layer, 
                           target_languages, save_dir=SAVE_DIR):
    """
    Sauvegarde les rankings de features par layer et par langue.
    
    Structure fichiers :
        sae_features/
            metadata.json              ← langues, layers, dimensions
            layer_0_indices.pt         ← top_index  (num_langs, d_sae)
            layer_0_values.pt          ← top_values (num_langs, d_sae)
            layer_5_indices.pt
            ...
            layer_0_readable.json      ← top-20 features par langue, lisible
            ...
    """
    # Métadonnées
    metadata = {
        "languages"      : target_languages,
        "layers"         : sorted(top_index_per_layer.keys()),
        "d_sae"          : top_index_per_layer[list(top_index_per_layer.keys())[0]].shape[-1],
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    for layer in top_index_per_layer:
        # Tensors bruts — pour recharger et utiliser directement dans le code
        torch.save(
            top_index_per_layer[layer],
            os.path.join(save_dir, f"layer_{layer}_indices.pt")
        )
        torch.save(
            top_values_per_layer[layer],
            os.path.join(save_dir, f"layer_{layer}_values.pt")
        )

        # Version lisible JSON — top-20 features par langue avec leur v-score
        readable = {}
        for i, lang in enumerate(target_languages):
            readable[lang] = [
                {
                    "rank"        : k + 1,
                    "feature_idx" : top_index_per_layer[layer][i, k].item(),
                    "v_score"     : round(top_values_per_layer[layer][i, k].item(), 6),
                }
                for k in range(20)  # top-20
            ]
        with open(os.path.join(save_dir, f"layer_{layer}_readable.json"), "w") as f:
            json.dump(readable, f, indent=2, ensure_ascii=False)

    print(f"✅ Sauvegardé dans '{save_dir}/'")
    print(f"   {len(top_index_per_layer)} layers × {len(target_languages)} langues")


# =============================================================================
# CHARGER
# =============================================================================

def load_feature_rankings(save_dir=SAVE_DIR):
    """
    Recharge les rankings sauvegardés.
    Retourne top_index_per_layer, top_values_per_layer, metadata
    """
    with open(os.path.join(save_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    top_index_per_layer  = {}
    top_values_per_layer = {}

    for layer in metadata["layers"]:
        top_index_per_layer[layer] = torch.load(
            os.path.join(save_dir, f"layer_{layer}_indices.pt"),
            weights_only=True
        )
        top_values_per_layer[layer] = torch.load(
            os.path.join(save_dir, f"layer_{layer}_values.pt"),
            weights_only=True
        )

    print(f"✅ Chargé depuis '{save_dir}/'")
    print(f"   Layers disponibles : {metadata['layers']}")
    print(f"   Langues : {metadata['languages']}")
    return top_index_per_layer, top_values_per_layer, metadata


# =============================================================================
# 4. SAE sanity check
# =============================================================================

print("\n[4] SAE sanity check...")
sae_probe = SAE.from_pretrained(SAE_RELEASE, "layer_18").to(device)
dummy = tokenizer.encode("Hello, world!", return_tensors='pt').to(device)
act   = gather_residual_activations(model, 18, dummy)
enc   = sae_probe.encode(act)
print(f"    d_model={act.shape[-1]}  d_sae={enc.shape[-1]}  W_dec={tuple(sae_probe.W_dec.shape)}")
del sae_probe


# =============================================================================
# USAGE — remplace la section [5] du code principal
# =============================================================================

RECOMPUTE = True  # ← mettre False après le premier run

if RECOMPUTE or not os.path.exists(os.path.join(SAVE_DIR, "metadata.json")):
    print("\n[5] Calcul des v-scores...")
    top_index_per_layer  = {}
    top_values_per_layer = {}

    for layer in ABLATION_LAYERS:
        print(f"    Layer {layer:2d}...")
        sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)
        idx, vals = compute_top_index_per_lan_for_layer(
            model, sae, layer, TARGET_LANGUAGES, multilingual_texts, device,
            n_texts_per_lan=N_TEXTS_PER_LAN,
        )
        top_index_per_layer[layer]  = idx
        top_values_per_layer[layer] = vals
        del sae

    save_feature_rankings(top_index_per_layer, top_values_per_layer, TARGET_LANGUAGES)

else:
    print("\n[5] Chargement des v-scores depuis le cache...")
    top_index_per_layer, top_values_per_layer, metadata = load_feature_rankings()

# # =============================================================================

# PLOT_LAYER = 18
# vals = top_values_per_layer[PLOT_LAYER]

# rows = []
# for i, lan in enumerate(TARGET_LANGUAGES):
#     for k in range(3):
#         rows.append({"Language": lan, "Rank": f"Top-{k+1}", "v score": vals[i, k].item()})

# plt.figure(figsize=(12, 5))
# sns.barplot(data=pd.DataFrame(rows), x="Language", y="v score", hue="Rank", palette="viridis")
# plt.title(f"Language-specificity score (v) — Layer {PLOT_LAYER}")
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("v_scores.png", dpi=150)
# plt.show()
# print("    Saved v_scores.png")

# # # =============================================================================
# # 6 & 7. Ablation on target language + plot
# # =============================================================================

# print(f"\n[6] Ablating {LANG_TO_ABLATE.upper()} features on {LANG_TO_ABLATE.upper()} text...")
# lang_idx       = TARGET_LANGUAGES.index(LANG_TO_ABLATE)
# lang_sentences = multilingual_texts[lang_idx * 100 : lang_idx * 100 + N_SENTENCES]

# delta_per_layer = {}
# for layer in ABLATION_LAYERS:
#     print(f"    Layer {layer:2d}...", end="  ")
#     sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)
#     result = run_ablation(
#         model, sae, layer, top_index_per_layer[layer],
#         lang_idx, lang_sentences, tokenizer, device,
#         n_langs=len(TARGET_LANGUAGES),
#     )
#     delta_per_layer[layer] = result
#     del sae
#     print({k: f"{v:+.3f}" for k, v in result.items()})

# layers_sorted = sorted(delta_per_layer.keys())
# config_names  = ["Ablate top-1", "Ablate top-2", "Ablate top-1 & 2"]
# colors        = ["steelblue", "darkorange", "seagreen"]

# plt.figure(figsize=(10, 5))
# for name, color in zip(config_names, colors):
#     deltas = [delta_per_layer[l][name] for l in layers_sorted]
#     plt.plot(layers_sorted, deltas, marker='o', label=name, color=color)
# plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
# plt.xlabel("Layer")
# plt.ylabel("Delta CE loss (ablated - baseline)")
# plt.title(f"Ablation — {LANG_TO_ABLATE.upper()} features on {LANG_TO_ABLATE.upper()} text")
# plt.legend()
# plt.grid(alpha=0.4)
# plt.tight_layout()
# plt.savefig(f"ablation_{LANG_TO_ABLATE}.png", dpi=150)
# plt.show()
# print(f"    Saved ablation_{LANG_TO_ABLATE}.png")


# # =============================================================================
# # 8. Cross-language specificity check
# # =============================================================================

# print(f"\n[8] Ablating {LANG_TO_ABLATE.upper()} features on {OTHER_LANGUAGE.upper()} text...")
# other_lang_idx  = TARGET_LANGUAGES.index(OTHER_LANGUAGE)
# other_sentences = multilingual_texts[other_lang_idx * 100 : other_lang_idx * 100 + N_SENTENCES]

# delta_other_per_layer = {}
# for layer in ABLATION_LAYERS:
#     print(f"    Layer {layer:2d}...", end="  ")
#     sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)
#     result = run_ablation(
#         model, sae, layer, top_index_per_layer[layer],
#         lang_idx,        # still French features
#         other_sentences, # but Japanese text
#         tokenizer, device, n_langs=len(TARGET_LANGUAGES),
#     )
#     delta_other_per_layer[layer] = result
#     del sae
#     print({k: f"{v:+.3f}" for k, v in result.items()})

# fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
# for ax, (delta_dict, title) in zip(axes, [
#     (delta_per_layer,       f"Evaluated on {LANG_TO_ABLATE.upper()} text"),
#     (delta_other_per_layer, f"Evaluated on {OTHER_LANGUAGE.upper()} text"),
# ]):
#     for name, color in zip(config_names, colors):
#         deltas = [delta_dict[l][name] for l in layers_sorted]
#         ax.plot(layers_sorted, deltas, marker='o', label=name, color=color)
#     ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
#     ax.set_xlabel("Layer")
#     ax.set_ylabel("Delta CE loss")
#     ax.set_title(f"Ablate {LANG_TO_ABLATE.upper()} features — {title}")
#     ax.legend(fontsize=8)
#     ax.grid(alpha=0.4)

# plt.suptitle(
#     f"{LANG_TO_ABLATE.upper()} features hurt {LANG_TO_ABLATE.upper()} more than {OTHER_LANGUAGE.upper()}",
#     fontsize=11
# )
# plt.tight_layout()
# plt.savefig("ablation_specificity.png", dpi=150)
# plt.show()
# print("    Saved ablation_specificity.png")


