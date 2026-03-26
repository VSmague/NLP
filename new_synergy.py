"""
cluster_synergy.py
==================
Analyse la synergie intra-cluster vs inter-cluster.

Pour chaque paire de langues (lang_A, lang_B) :
  - Ablate top-1 feature de lang_A seule  → ΔCE_A
  - Ablate top-1 feature de lang_B seule  → ΔCE_B
  - Ablate top-1 de A + top-1 de B ensemble → ΔCE_AB
  - Synergy = ΔCE_AB - (ΔCE_A + ΔCE_B)

Évalué sur la langue cible (lang_A) à chaque layer.

Hypothèse : synergie > 0 pour paires intra-cluster,
            synergie ≈ 0 pour paires inter-cluster.

Requires: sae_features/ with ablation.py outputs.
"""

import os
import json
import pickle
import configparser
import time
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# ── Auth ──────────────────────────────────────────────────────────────────────
config = configparser.ConfigParser()
config.read("secrets.ini")
os.environ["HF_TOKEN"] = config["huggingface"]["token"]

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-0.6B"
SAE_RELEASE = "mwhanna-qwen3-0.6b-transcoders-lowl0"
SAVE_DIR    = "sae_features"
OUTPUT_DIR  = "plots_synergy_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_PATH  = os.path.join(SAVE_DIR, "cluster_synergy_cache.pkl")
N_SENTENCES = 100
RECOMPUTE   = True

# ── Clusters (from cross_linguistic_interaction.py output) ────────────────────
CLUSTERS = {
    1: ['th'],
    2: ['ar'],
    3: ['en', 'es', 'fr', 'pt', 'vi'],
    4: ['ja', 'zh'],
    5: ['ko'],
}

# Build lang → cluster_id mapping
LANG_TO_CLUSTER = {
    lang: cid
    for cid, langs in CLUSTERS.items()
    for lang in langs
}

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'ja': 'Japanese', 'ko': 'Korean', 'pt': 'Portuguese',
    'th': 'Thai', 'vi': 'Vietnamese', 'zh': 'Chinese', 'ar': 'Arabic',
}

CLUSTER_COLORS = {
    1: '#64B5CD',   # TH  — Kra-Dai
    2: '#C44E52',   # AR  — Semitic
    3: '#55A868',   # EN/ES/FR/PT/VI — Indo-European + Austroasiatic
    4: '#CCB974',   # JA/ZH — East Asian
    5: '#8172B3',   # KO  — Koreanic
}

# ── All pairs (A, B) with A ≠ B ───────────────────────────────────────────────
# We evaluate ΔCE on lang_A's text
ALL_PAIRS = [
    (a, b)
    for a in TARGET_LANGUAGES
    for b in TARGET_LANGUAGES
    if a != b
]

# Categorise each pair
def pair_type(a, b):
    return "intra" if LANG_TO_CLUSTER[a] == LANG_TO_CLUSTER[b] else "inter"

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================================================
# Utility functions
# =============================================================================

def fmt_time(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

def directional_ablation_forward(model, sae_a, layer_a, feat_a,
                                  sae_b, layer_b, feat_b, inputs):
    """Apply two ablation hooks simultaneously (same layer, different features)."""
    W_dec_a = sae_a.W_dec.T.to(device)
    dirs_a  = W_dec_a[:, feat_a]
    norms_a = torch.norm(dirs_a, dim=0) ** 2
    dn_a    = dirs_a / norms_a.clamp(min=1e-8)

    W_dec_b = sae_b.W_dec.T.to(device)
    dirs_b  = W_dec_b[:, feat_b]
    norms_b = torch.norm(dirs_b, dim=0) ** 2
    dn_b    = dirs_b / norms_b.clamp(min=1e-8)

    # Both SAEs target the same layer (top-1 of A and top-1 of B at layer_a == layer_b)
    def _hook(module, inp, output):
        act = (output[0] if isinstance(output, tuple) else output).to(torch.float32)
        act = act - (act @ dn_a) @ dn_a.T
        act = act - (act @ dn_b) @ dn_b.T
        return (act.to(output[0].dtype),) + output[1:] if isinstance(output, tuple) \
               else act.to(output.dtype)

    handle = model.model.layers[layer_a].register_forward_hook(_hook)
    try:
        out = model.forward(inputs)
    finally:
        handle.remove()
    return out


def ablate_single(model, sae, layer, feat_idx, inputs):
    W_dec     = sae.W_dec.T.to(device)
    dirs      = W_dec[:, feat_idx]
    norms     = torch.norm(dirs, dim=0) ** 2
    dirs_norm = dirs / norms.clamp(min=1e-8)

    def _hook(module, inp, output):
        act = (output[0] if isinstance(output, tuple) else output).to(torch.float32)
        act = act - (act @ dirs_norm) @ dirs_norm.T
        return (act.to(output[0].dtype),) + output[1:] if isinstance(output, tuple) \
               else act.to(output.dtype)

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        out = model.forward(inputs)
    finally:
        handle.remove()
    return out


def ce_loss(logits, inputs):
    sl = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    tl = inputs[:, 1:].contiguous().view(-1)
    return F.cross_entropy(sl, tl)


def mean_delta(model, sae, layer, feat, sentences, tokenizer,
               sae_b=None, feat_b=None):
    """
    If sae_b is None : single ablation.
    Else             : joint ablation of feat (from sae) and feat_b (from sae_b).
    """
    deltas = []
    for text in sentences:
        inp = tokenizer.encode(text, return_tensors='pt',
                               add_special_tokens=True).to(device)
        if inp.size(1) < 2:
            continue
        base = ce_loss(model.forward(inp)["logits"], inp).item()
        if sae_b is None:
            abl = ce_loss(ablate_single(model, sae, layer, feat, inp)["logits"], inp).item()
        else:
            abl = ce_loss(
                directional_ablation_forward(
                    model, sae, layer, feat, sae_b, layer, feat_b, inp
                )["logits"], inp
            ).item()
        deltas.append(abl - base)
    return float(np.nanmean(deltas)) if deltas else 0.0


# =============================================================================
# Load SAE cache
# =============================================================================

print("\n[0] Loading SAE cache...")
with open(os.path.join(SAVE_DIR, "metadata.json")) as f:
    meta = json.load(f)
LAYERS = meta["layers"]

top_index_all = {}
for layer in LAYERS:
    top_index_all[layer] = torch.load(
        os.path.join(SAVE_DIR, f"layer_{layer}_indices.pt"), weights_only=True
    )
print(f"   {len(LAYERS)} layers: {LAYERS}")


# =============================================================================
# Compute or load synergy data
# =============================================================================
# synergy_data[(lang_a, lang_b)][layer_idx] = {
#   'delta_a'  : float,   # ΔCE ablate A only, eval on A
#   'delta_b'  : float,   # ΔCE ablate B only, eval on A
#   'delta_ab' : float,   # ΔCE ablate A+B,    eval on A
#   'synergy'  : float,   # delta_ab - (delta_a + delta_b)
# }

if RECOMPUTE or not os.path.exists(CACHE_PATH):
    print("\n[1] Loading model...")
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
    texts = []
    for line in resp.text.strip().split('\n'):
        obj = json.loads(line)
        texts.append(obj['text'] if isinstance(obj, dict) else obj)

    sentences_per_lang = {
        lang: texts[i * 100: i * 100 + N_SENTENCES]
        for i, lang in enumerate(TARGET_LANGUAGES)
    }
    print(f"   {len(texts)} sentences loaded")

    print(f"\n[2] Computing synergy for {len(ALL_PAIRS)} pairs × {len(LAYERS)} layers...")

    synergy_data = {}
    t0    = time.time()
    total = len(ALL_PAIRS) * len(LAYERS)
    done  = 0

    for (lang_a, lang_b) in ALL_PAIRS:
        j_a = TARGET_LANGUAGES.index(lang_a)
        j_b = TARGET_LANGUAGES.index(lang_b)
        sents_a = sentences_per_lang[lang_a]
        synergy_data[(lang_a, lang_b)] = []

        for layer in LAYERS:
            sae_a = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)
            sae_b = SAE.from_pretrained(SAE_RELEASE, f"layer_{layer}").to(device)

            feat_a = top_index_all[layer][j_a, 0:1].to(device)  # top-1 of lang_a
            feat_b = top_index_all[layer][j_b, 0:1].to(device)  # top-1 of lang_b

            da  = mean_delta(model, sae_a, layer, feat_a, sents_a, tokenizer)
            db  = mean_delta(model, sae_b, layer, feat_b, sents_a, tokenizer)
            dab = mean_delta(model, sae_a, layer, feat_a, sents_a, tokenizer,
                             sae_b=sae_b, feat_b=feat_b)
            syn = dab - (da + db)

            synergy_data[(lang_a, lang_b)].append({
                'delta_a'  : da,
                'delta_b'  : db,
                'delta_ab' : dab,
                'synergy'  : syn,
            })

            del sae_a, sae_b
            done += 1
            elapsed = time.time() - t0
            eta     = (elapsed / done) * (total - done)
            pct     = done / total * 100
            print(f"\r   [{lang_a.upper()}+{lang_b.upper()} | L{layer:02d}] "
                  f"{pct:5.1f}%  ETA {fmt_time(eta)}", end="", flush=True)

    print()
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            'synergy_data': synergy_data,
            'layers'      : LAYERS,
            'clusters'    : CLUSTERS,
        }, f)
    print(f"✅ Saved to {CACHE_PATH}")

else:
    print(f"\n[1-2] Loading from cache ({CACHE_PATH})...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    synergy_data = cache['synergy_data']
    LAYERS       = cache['layers']
    print(f"   ✅ {len(synergy_data)} pairs × {len(LAYERS)} layers")


# =============================================================================
# Aggregate synergy per layer: intra vs inter cluster
# =============================================================================

n_layers = len(LAYERS)

intra_synergy_per_layer = [[] for _ in range(n_layers)]
inter_synergy_per_layer = [[] for _ in range(n_layers)]

for (lang_a, lang_b), layer_vals in synergy_data.items():
    pt = pair_type(lang_a, lang_b)
    for li, d in enumerate(layer_vals):
        if pt == "intra":
            intra_synergy_per_layer[li].append(d['synergy'])
        else:
            inter_synergy_per_layer[li].append(d['synergy'])

intra_mean = np.array([np.mean(v) if v else 0 for v in intra_synergy_per_layer])
intra_std  = np.array([np.std(v)  if v else 0 for v in intra_synergy_per_layer])
inter_mean = np.array([np.mean(v) if v else 0 for v in inter_synergy_per_layer])
inter_std  = np.array([np.std(v)  if v else 0 for v in inter_synergy_per_layer])


# =============================================================================
# PLOT 1 — Intra vs inter synergy per layer
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(LAYERS, intra_mean, color='#C44E52', linewidth=2.2, marker='o',
        markersize=5, label='Intra-cluster pairs')
ax.fill_between(LAYERS,
                intra_mean - intra_std,
                intra_mean + intra_std,
                color='#C44E52', alpha=0.15)

ax.plot(LAYERS, inter_mean, color='#4C72B0', linewidth=2.2, marker='s',
        markersize=5, label='Inter-cluster pairs')
ax.fill_between(LAYERS,
                inter_mean - inter_std,
                inter_mean + inter_std,
                color='#4C72B0', alpha=0.15)

ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("Mean synergy  ΔCE_AB − (ΔCE_A + ΔCE_B)", fontsize=10)
ax.set_title("Intra-cluster vs inter-cluster feature synergy per layer\n"
             "(ablate top-1 of lang A + top-1 of lang B, evaluate on lang A)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_synergy_intra_vs_inter_per_layer.png"), dpi=150)
plt.close()
print("→ 1_synergy_intra_vs_inter_per_layer.png")


# =============================================================================
# PLOT 2 — Per-cluster intra synergy heatmap (avg over layers)
# =============================================================================

# Average synergy over layers for each pair
pair_avg_synergy = {
    pair: np.mean([d['synergy'] for d in vals])
    for pair, vals in synergy_data.items()
}

# Build n×n matrix
n  = len(TARGET_LANGUAGES)
S  = np.zeros((n, n))
for (a, b), val in pair_avg_synergy.items():
    i = TARGET_LANGUAGES.index(a)
    j = TARGET_LANGUAGES.index(b)
    S[i, j] = val

lang_labels = [LANG_NAMES[l] for l in TARGET_LANGUAGES]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(S, cmap='RdBu_r', aspect='auto',
               vmin=-np.abs(S).max(), vmax=np.abs(S).max())
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(lang_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(lang_labels, fontsize=9)
ax.set_xlabel("Feature source: language B (top-1 ablated)", fontsize=10)
ax.set_ylabel("Evaluation language A (text + feature A ablated)", fontsize=10)

for i in range(n):
    for j in range(n):
        if i == j:
            ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                                        fill=True, color='#e0e0e0', zorder=0))
        ax.text(j, i, f"{S[i,j]:+.3f}", ha='center', va='center',
                fontsize=7, color='black' if abs(S[i,j]) < np.abs(S).max()*0.6 else 'white')

# Draw cluster borders
cluster_boundaries = []
count = 0
for cid in sorted(CLUSTERS.keys()):
    count += len(CLUSTERS[cid])
    cluster_boundaries.append(count)

# Reorder axes by cluster for clarity
cluster_order = [
    TARGET_LANGUAGES.index(l)
    for cid in sorted(CLUSTERS.keys())
    for l in CLUSTERS[cid]
]
S_ordered = S[np.ix_(cluster_order, cluster_order)]
labels_ordered = [LANG_NAMES[TARGET_LANGUAGES[k]] for k in cluster_order]

fig2, ax2 = plt.subplots(figsize=(10, 8))
im2 = ax2.imshow(S_ordered, cmap='RdBu_r', aspect='auto',
                  vmin=-np.abs(S).max(), vmax=np.abs(S).max())
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(labels_ordered, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(labels_ordered, fontsize=9)
ax2.set_xlabel("Feature source: language B (top-1 ablated)", fontsize=10)
ax2.set_ylabel("Evaluation language A", fontsize=10)

# Annotate cells
for i in range(n):
    for j in range(n):
        ax2.text(j, i, f"{S_ordered[i,j]:+.3f}", ha='center', va='center',
                 fontsize=7, color='black' if abs(S_ordered[i,j]) < np.abs(S).max()*0.6 else 'white')

# Draw cluster block borders
pos = -0.5
for cid in sorted(CLUSTERS.keys()):
    size = len(CLUSTERS[cid])
    rect = plt.Rectangle((pos, pos), size, size,
                          fill=False, edgecolor=CLUSTER_COLORS[cid],
                          linewidth=2.5, zorder=5)
    ax2.add_patch(rect)
    ax2.text(pos + size/2, pos - 0.6, f"C{cid}",
             ha='center', va='bottom', fontsize=8,
             color=CLUSTER_COLORS[cid], fontweight='bold')
    pos += size

plt.colorbar(im2, ax=ax2, label="Mean synergy (avg. all layers)")
ax2.set_title("Feature synergy matrix — ordered by cluster\n"
              "Diagonal blocks = intra-cluster (expected positive synergy)",
              fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_synergy_matrix_by_cluster.png"), dpi=150)
plt.close()
print("→ 2_synergy_matrix_by_cluster.png")


# =============================================================================
# PLOT 3 — Example intra-cluster per layer: Cluster 3 (FR, ES, PT, EN, VI)
# =============================================================================

CLUSTER3_LANGS = CLUSTERS[3]   # ['en', 'es', 'fr', 'pt', 'vi']

# All intra-cluster pairs within cluster 3, evaluated on lang_a
intra_pairs_c3 = [
    (a, b) for a in CLUSTER3_LANGS for b in CLUSTER3_LANGS if a != b
]

fig, axes = plt.subplots(
    len(CLUSTER3_LANGS), len(CLUSTER3_LANGS) - 1,
    figsize=(4 * (len(CLUSTER3_LANGS)-1), 3.5 * len(CLUSTER3_LANGS)),
    sharey=False
)

for row, lang_a in enumerate(CLUSTER3_LANGS):
    col_idx = 0
    for lang_b in CLUSTER3_LANGS:
        if lang_b == lang_a:
            continue
        ax = axes[row][col_idx]
        vals = synergy_data.get((lang_a, lang_b), None)
        if vals is None:
            ax.axis('off')
            col_idx += 1
            continue

        da_vals  = [d['delta_a']  for d in vals]
        db_vals  = [d['delta_b']  for d in vals]
        dab_vals = [d['delta_ab'] for d in vals]
        syn_vals = [d['synergy']  for d in vals]

        ax.plot(LAYERS, da_vals,  color='#4C72B0', linewidth=1.5,
                linestyle='-',  marker='.', markersize=4,
                label=f"Ablate {lang_a.upper()} top-1")
        ax.plot(LAYERS, db_vals,  color='#DD8452', linewidth=1.5,
                linestyle='--', marker='.', markersize=4,
                label=f"Ablate {lang_b.upper()} top-1")
        ax.plot(LAYERS, dab_vals, color='#C44E52', linewidth=2.0,
                linestyle='-',  marker='o', markersize=4,
                label=f"Ablate both")
        ax.fill_between(LAYERS,
                        [a + b for a, b in zip(da_vals, db_vals)],
                        dab_vals,
                        alpha=0.2, color='#C44E52',
                        label="Synergy region")
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.7)
        ax.set_title(f"Eval: {LANG_NAMES[lang_a]}\nB: {LANG_NAMES[lang_b]}",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_ylabel("ΔCE", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row == 0 and col_idx == 0:
            ax.legend(fontsize=5.5, loc='upper left', framealpha=0.5)
        col_idx += 1

fig.suptitle(
    "Cluster 3 (EN/ES/FR/PT/VI) — intra-cluster synergy per layer\n"
    "Red region = synergy (ΔCE_AB > ΔCE_A + ΔCE_B)",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_synergy_cluster3_detail.png"), dpi=150)
plt.close()
print("→ 3_synergy_cluster3_detail.png")


# =============================================================================
# PLOT 4 — Intra-cluster synergy averaged per cluster (bar chart)
# =============================================================================

cluster_intra_syn = {}
for cid, langs in CLUSTERS.items():
    pairs = [(a, b) for a in langs for b in langs if a != b]
    if not pairs:
        cluster_intra_syn[cid] = (0.0, 0.0)
        continue
    all_syn = [
        d['synergy']
        for (a, b) in pairs
        for d in synergy_data.get((a, b), [])
    ]
    cluster_intra_syn[cid] = (np.mean(all_syn), np.std(all_syn))

cluster_inter_syn = {}
for cid, langs in CLUSTERS.items():
    other_langs = [l for l in TARGET_LANGUAGES if l not in langs]
    pairs = [(a, b) for a in langs for b in other_langs]
    if not pairs:
        cluster_inter_syn[cid] = (0.0, 0.0)
        continue
    all_syn = [
        d['synergy']
        for (a, b) in pairs
        for d in synergy_data.get((a, b), [])
    ]
    cluster_inter_syn[cid] = (np.mean(all_syn), np.std(all_syn))

cids   = sorted(CLUSTERS.keys())
x      = np.arange(len(cids))
width  = 0.35
labels_c = [f"C{c}\n({', '.join(l.upper() for l in CLUSTERS[c])})" for c in cids]

fig, ax = plt.subplots(figsize=(10, 5))
intra_means = [cluster_intra_syn[c][0] for c in cids]
intra_stds  = [cluster_intra_syn[c][1] for c in cids]
inter_means = [cluster_inter_syn[c][0] for c in cids]
inter_stds  = [cluster_inter_syn[c][1] for c in cids]

bars1 = ax.bar(x - width/2, intra_means, width, yerr=intra_stds,
               color=[CLUSTER_COLORS[c] for c in cids], alpha=0.85,
               capsize=4, label='Intra-cluster synergy', edgecolor='white')
bars2 = ax.bar(x + width/2, inter_means, width, yerr=inter_stds,
               color=[CLUSTER_COLORS[c] for c in cids], alpha=0.35,
               capsize=4, label='Inter-cluster synergy', edgecolor='grey',
               hatch='//')

ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax.set_xticks(x)
ax.set_xticklabels(labels_c, fontsize=9)
ax.set_ylabel("Mean synergy  ΔCE_AB − (ΔCE_A + ΔCE_B)", fontsize=10)
ax.set_title("Intra-cluster vs inter-cluster synergy per cluster\n"
             "(solid = intra, hatched = inter; avg over all layers)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_synergy_per_cluster_bar.png"), dpi=150)
plt.close()
print("→ 4_synergy_per_cluster_bar.png")


# =============================================================================
# PLOT 5 — FR + ES example: ablate top-1 FR / top-1 ES / both, eval on FR
#          Mirrors the Figure 6 style of reproduce_figures_section5.py
# =============================================================================

EXAMPLE_PAIRS = [
    ("fr", "es", "Intra-cluster: FR eval, ablate ES (C3)"),
    ("fr", "pt", "Intra-cluster: FR eval, ablate PT (C3)"),
    ("fr", "ar", "Inter-cluster: FR eval, ablate AR (C2)"),
    ("fr", "ja", "Inter-cluster: FR eval, ablate JA (C4)"),
    ("ja", "zh", "Intra-cluster: JA eval, ablate ZH (C4)"),
    ("ja", "ko", "Inter-cluster: JA eval, ablate KO (C5)"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (lang_a, lang_b, title) in zip(axes, EXAMPLE_PAIRS):
    vals = synergy_data.get((lang_a, lang_b), None)
    if vals is None:
        ax.axis('off')
        continue

    da_vals  = [d['delta_a']  for d in vals]
    db_vals  = [d['delta_b']  for d in vals]
    dab_vals = [d['delta_ab'] for d in vals]
    additive = [a + b for a, b in zip(da_vals, db_vals)]

    ax.plot(LAYERS, da_vals,  color='#4C72B0', linewidth=1.8, linestyle='-',
            marker='.', markersize=5,
            label=f"Ablate {LANG_NAMES[lang_a]} top-1 only")
    ax.plot(LAYERS, db_vals,  color='#DD8452', linewidth=1.8, linestyle='--',
            marker='.', markersize=5,
            label=f"Ablate {LANG_NAMES[lang_b]} top-1 only")
    ax.plot(LAYERS, additive, color='#888888', linewidth=1.2, linestyle=':',
            label="Sum (A + B) — expected if additive")
    ax.plot(LAYERS, dab_vals, color='#C44E52', linewidth=2.4, linestyle='-',
            marker='o', markersize=5,
            label="Ablate both (A + B)")

    ax.fill_between(LAYERS, additive, dab_vals,
                    where=[d > a for d, a in zip(dab_vals, additive)],
                    alpha=0.25, color='#C44E52', label="Positive synergy")
    ax.fill_between(LAYERS, additive, dab_vals,
                    where=[d < a for d, a in zip(dab_vals, additive)],
                    alpha=0.20, color='#4C72B0', label="Negative synergy")

    ax.axhline(0, color='grey', linestyle=':', linewidth=0.7)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("ΔCE (eval on " + LANG_NAMES[lang_a] + ")", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,
           framealpha=0.5, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(
    "Intra vs inter-cluster feature synergy — selected pairs\n"
    "Red fill = super-additive (synergy); blue fill = sub-additive",
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "5_synergy_example_pairs.png"), dpi=150)
plt.close()
print("→ 5_synergy_example_pairs.png")


print(f"\n✅ All plots saved to '{OUTPUT_DIR}/'  —  5 plots")
print("   Intra-cluster synergy expected to be higher than inter-cluster.")