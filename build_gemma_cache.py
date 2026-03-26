"""
build_gemma_sae_cache.py
=========================
Converts v_scores_run_reprod_fig_1_top5.csv into the .pt / metadata.json
files expected by reproduce_figures_section5.py (Gemma version).

The CSV has columns:
  layer, language_code, rank, feature_index, nu_value, ...

Output (mirrors what ablation.py produces for Qwen):
  sae_features_gemma/
    metadata.json                   ← languages list + layers list
    layer_0_indices.pt              ← LongTensor shape (n_langs, max_rank)
    layer_2_indices.pt
    ...

NOTE: The CSV contains 'ks' (Kashmiri) but NOT 'ar' (Arabic).
  → By default this script maps 'ks' → replaces it with a dummy 'ar' row
    using the same feature indices as 'ks' (both are Arabic-script languages,
    so the top SAE features are likely shared).
  → Set TREAT_KS_AS_AR = False to keep 'ks' as a separate language instead
    and drop 'ar' from TARGET_LANGUAGES.
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "v_scores_run_reprod_fig_1_top5.csv"
SAVE_DIR   = "sae_features_gemma"
TOP_K      = 5          # number of ranked features per lang/layer in the CSV
TREAT_KS_AS_AR = True   # map Kashmiri ('ks') → Arabic ('ar') placeholder

os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']

# =============================================================================
# Load CSV
# =============================================================================

print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows  |  layers: {sorted(df.layer.unique())}  "
      f"|  langs: {sorted(df.language_code.unique())}")

# Map ks → ar if requested
if TREAT_KS_AS_AR:
    n_ks = (df.language_code == 'ks').sum()
    df.loc[df.language_code == 'ks', 'language_code'] = 'ar'
    print(f"  Mapped {n_ks} 'ks' rows → 'ar'")

# Verify all target languages are now present
missing = [l for l in TARGET_LANGUAGES if l not in df.language_code.unique()]
if missing:
    raise ValueError(
        f"Languages missing from CSV after remapping: {missing}\n"
        f"Set TREAT_KS_AS_AR=True or adjust TARGET_LANGUAGES."
    )

LAYERS = sorted(df.layer.unique().tolist())
n_langs = len(TARGET_LANGUAGES)
print(f"  {n_langs} languages  |  {len(LAYERS)} layers: {LAYERS}")


# =============================================================================
# Build index tensors
# =============================================================================
# For each layer: LongTensor of shape (n_langs, TOP_K)
# Row i = feature indices for TARGET_LANGUAGES[i], sorted by rank ascending

for layer in LAYERS:
    df_layer = df[df.layer == layer]
    indices  = np.zeros((n_langs, TOP_K), dtype=np.int64)

    for i, lang in enumerate(TARGET_LANGUAGES):
        df_lang = (df_layer[df_layer.language_code == lang]
                   .sort_values('rank')
                   .head(TOP_K))
        if len(df_lang) == 0:
            print(f"  WARNING: no data for lang={lang} layer={layer}")
            continue
        feat_idx = df_lang['feature_index'].values
        # Pad with -1 if fewer than TOP_K features available
        n = min(len(feat_idx), TOP_K)
        indices[i, :n] = feat_idx[:n]
        if n < TOP_K:
            indices[i, n:] = feat_idx[-1]   # repeat last index as fallback

    tensor = torch.tensor(indices, dtype=torch.long)
    out_path = os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt")
    torch.save(tensor, out_path)
    print(f"  Saved layer_{layer}_indices.pt  shape={tuple(tensor.shape)}")

    # Quick sanity check: print rank-1 feature for each lang at this layer
    print(f"    Rank-1 features at layer {layer}:")
    for i, lang in enumerate(TARGET_LANGUAGES):
        print(f"      {lang.upper():3s}  feature {indices[i, 0]}")


# =============================================================================
# Write metadata.json
# =============================================================================

metadata = {
    "model"    : "google/gemma-2-2b",
    "languages": TARGET_LANGUAGES,
    "layers"   : LAYERS,
    "top_k"    : TOP_K,
    "source"   : CSV_PATH,
    "ks_mapped_to_ar": TREAT_KS_AS_AR,
}

meta_path = os.path.join(SAVE_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\nSaved {meta_path}")
print(json.dumps(metadata, indent=2))

print(f"\n✅ sae_features_gemma/ ready — {len(LAYERS)} layers × {n_langs} languages")
print("   Next step: run reproduce_figures_section5.py (Gemma version)")