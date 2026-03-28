"""
compute_nu_scores.py
====================
Computes language-specificity (nu) scores for each SAE feature at each layer
and saves the rankings to disk.

    nu_i(f) = mean_act_lang_i(f) - mean_{j != i} mean_act_lang_j(f)

Output (in sae_features/):
    metadata.json            — languages, layers, d_sae
    layer_X_indices.pt       — feature indices sorted by nu desc  (n_langs, d_sae)
    layer_X_values.pt        — corresponding nu values            (n_langs, d_sae)
    layer_X_readable.json    — top-20 features per language, human-readable

Set RECOMPUTE = False after the first run to load from cache.
"""

import os
import json
import configparser
import numpy as np
import torch
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# ── Auth ──────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("secrets.ini")
os.environ["HF_TOKEN"] = config["huggingface"]["token"]

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-0.6B"
SAE_RELEASE      = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR         = "sae_features"
ABLATION_LAYERS  = range(28)
N_TEXTS_PER_LANG = 100
RECOMPUTE        = False

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']

os.makedirs(SAVE_DIR, exist_ok=True)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================================================
# Core: nu score computation
# =============================================================================

def gather_residual_activations(model, layer, inputs):
    """Return the residual-stream output of a given layer. Shape: (1, seq, d_model)."""
    target_act = None

    def _hook(mod, inp, out):
        nonlocal target_act
        target_act = out[0]

    handle = model.model.layers[layer].register_forward_hook(_hook)
    model.forward(inputs)
    handle.remove()
    return target_act


def compute_nu_scores_for_layer(model, sae, layer, multilingual_texts, device):
    """
    Compute nu scores for all features at a given layer.

    Returns
    -------
    top_indices : LongTensor  (n_langs, d_sae)  — feature indices sorted by nu desc
    top_values  : FloatTensor (n_langs, d_sae)  — corresponding nu values
    """
    # Collect mean SAE activations per language
    avg_act_per_lang = []
    for i, lang in enumerate(TARGET_LANGUAGES):
        lang_texts = multilingual_texts[i * 100 : i * 100 + N_TEXTS_PER_LANG]
        activations = []
        for text in lang_texts:
            inp = tokenizer.encode(text, return_tensors='pt',
                                   add_special_tokens=True).to(device)
            act = gather_residual_activations(model, layer, inp)  # (1, seq, d_model)
            sae_act = sae.encode(act).cpu()                       # (1, seq, d_sae)
            if sae_act.dim() == 2:
                sae_act = sae_act.unsqueeze(0)
            activations.append(sae_act)

        stacked = torch.cat(activations, dim=1)       # (1, total_tokens, d_sae)
        avg_act_per_lang.append(stacked.mean(dim=1).squeeze(0))  # (d_sae,)

    avg_act_per_lang = torch.stack(avg_act_per_lang)  # (n_langs, d_sae)

    # nu_i(f) = mean_act_i(f) - mean_{j != i} mean_act_j(f)
    top_indices, top_values = [], []
    for i in range(len(TARGET_LANGUAGES)):
        others = torch.cat([avg_act_per_lang[:i], avg_act_per_lang[i+1:]])
        nu = avg_act_per_lang[i] - others.mean(0)
        sorted_vals, sorted_idx = torch.sort(nu, descending=True)
        top_indices.append(sorted_idx)
        top_values.append(sorted_vals)

    return torch.stack(top_indices), torch.stack(top_values)


# =============================================================================
# Save / load
# =============================================================================

def save_rankings(top_index_per_layer, top_values_per_layer):
    metadata = {
        "languages" : TARGET_LANGUAGES,
        "layers"    : sorted(top_index_per_layer.keys()),
        "d_sae"     : top_index_per_layer[
                          list(top_index_per_layer.keys())[0]
                      ].shape[-1],
    }
    with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    for layer, indices in top_index_per_layer.items():
        torch.save(indices,
                   os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"))
        torch.save(top_values_per_layer[layer],
                   os.path.join(SAVE_DIR, f"layer_{layer}_values.pt"))

        # Human-readable top-20 per language
        readable = {
            lang: [
                {
                    "rank"       : k + 1,
                    "feature_idx": indices[i, k].item(),
                    "nu_score"   : round(top_values_per_layer[layer][i, k].item(), 6),
                }
                for k in range(20)
            ]
            for i, lang in enumerate(TARGET_LANGUAGES)
        }
        with open(os.path.join(SAVE_DIR, f"layer_{layer}_readable.json"),
                  "w", encoding="utf-8") as f:
            json.dump(readable, f, indent=2, ensure_ascii=False)

    print(f"Saved to '{SAVE_DIR}/'  "
          f"({len(top_index_per_layer)} layers × {len(TARGET_LANGUAGES)} languages)")


def load_rankings():
    with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    top_index_per_layer  = {}
    top_values_per_layer = {}
    for layer in metadata["layers"]:
        top_index_per_layer[layer] = torch.load(
            os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"), weights_only=True)
        top_values_per_layer[layer] = torch.load(
            os.path.join(SAVE_DIR, f"layer_{layer}_values.pt"), weights_only=True)

    print(f"Loaded from '{SAVE_DIR}/'")
    print(f"  Layers : {metadata['layers']}")
    print(f"  Languages : {metadata['languages']}")
    return top_index_per_layer, top_values_per_layer, metadata


# =============================================================================
# Main
# =============================================================================

if RECOMPUTE or not os.path.exists(os.path.join(SAVE_DIR, "metadata.json")):

    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    print(f"    {len(model.model.layers)} layers")

    print("\n[2] Loading multilingual data...")
    DATA_URL = ("https://raw.githubusercontent.com/Aatrox103/"
                "multilingual-llm-features/main/SAE/data/multilingual_data.jsonl")
    resp = requests.get(DATA_URL)
    resp.raise_for_status()
    multilingual_texts = []
    for line in resp.text.strip().split('\n'):
        obj = json.loads(line)
        multilingual_texts.append(obj['text'] if isinstance(obj, dict) else obj)
    print(f"    {len(multilingual_texts)} sentences")

    print("\n[3] Computing nu scores...")
    top_index_per_layer  = {}
    top_values_per_layer = {}

    for layer in ABLATION_LAYERS:
        print(f"    Layer {layer:2d}...", end=" ", flush=True)
        sae = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)
        idx, vals = compute_nu_scores_for_layer(
            model, sae, layer, multilingual_texts, device
        )
        top_index_per_layer[layer]  = idx
        top_values_per_layer[layer] = vals
        del sae
        print("done")

    save_rankings(top_index_per_layer, top_values_per_layer)

else:
    print("\nLoading from cache...")
    top_index_per_layer, top_values_per_layer, metadata = load_rankings()