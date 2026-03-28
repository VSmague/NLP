# Unveiling Language-Specific Features in LLMs via Sparse Autoencoders

Reproduction and extension of [Deng et al., ACL 2025](https://arxiv.org/abs/2402.16694) on **Qwen3-0.6B**, investigating whether language-specific SAE features constitute a specific, causal, and task-useful representation of language.

> **Model:** Qwen3-0.6B
> **SAE:** `mwhanna-qwen3-0.6b-transcoders-lowl0`


## Pipeline

Run scripts in the following order:

```
compute_nu_scores.py
        ↓
reproduce_figures_section5.py   ←→   cross_linguistic_interactions.py
        ↓                                       ↓
cluster_synergy.py              ←   (uses fig6_cache + interaction matrix)
        ↓
downstream_ablation.py
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `compute_nu_scores.py` | Computes ν-scores for all SAE features at each layer. Saves `layer_X_indices.pt` and `layer_X_values.pt` to `sae_features/`. Must run first. |
| `reproduce_figures_section5.py` | Reproduces Figures 5 & 6 from the paper. Fig 5: layer-wise ΔCE specificity per language. Fig 6: top-1 / top-2 / top-1+2 synergy for each language. |
| `cross_linguistic_interactions.py` | Builds the 10×10 cross-linguistic interaction matrix (mean ΔCE on language i when ablating features of language j). Clusters languages via MDS + Ward linkage (k=5). |
| `cluster_synergy.py` | Tests intra- vs inter-cluster feature synergy by ablating top-1 features of two languages simultaneously. |
| `downstream_ablation.py` | Evaluates the causal impact of top-1+2 ablation at layer 10 on two downstream tasks: QA (XQuAD, F1 + EM) and Sentiment Classification (Amazon Reviews, Accuracy + QWK, stratified). |
| `plot_sae_features.py` | Additional visualisations of SAE feature rankings and ν-score distributions. |
| `check_data.py` | Sanity checks on dataset loading and label distributions. |

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `secrets.ini` file at the root:

```ini
[huggingface]
token = hf_...
```

A HuggingFace account with access to the Gemma model gates is required for Gemma experiments ([request access here](https://huggingface.co/google/gemma-2-2b)).

---

## Outputs

| Directory | Contents |
|-----------|----------|
| `sae_features/` | Cached ν-score rankings (`.pt`), interaction matrix, downstream results |
| `output/` | Figures 5 & 6, cross-linguistic interaction plots, cluster maps, downstream plots |
| `SNLP/qwen06b_local/` | Local Qwen3-0.6B model weights |

---

## Key Results

- **Layer 10** is the dominant language-specificity layer in Qwen3-0.6B, with ΔCE peaking sharply for most languages. Korean and Thai deviate, peaking at layer 0.
- Ablation of just **2 features** at layer 10 causes near-complete collapse on QA (English −41.1 F1, Vietnamese −34.8, Chinese −28.9) and classification (French −85.5 pp accuracy, ΔQWK ≈ −0.54).
- **Arabic** is consistently robust to ablation across all tasks, consistent with its isolated position in the cross-linguistic interaction space.
- **Intra-cluster feature pairs are sub-additive**, not superadditive — ablating two same-cluster language features together produces less effect than the sum of individual ablations, interpreted as evidence of shared feature directions within clusters.
- **Vietnamese** (Austroasiatic) clusters with European languages (EN/ES/FR/PT) in the interaction space rather than with Thai or Korean, suggesting the clustering reflects feature-space overlap in the model rather than typological proximity.

---

## Reference

```bibtex
@inproceedings{deng2025unveiling,
  title     = {Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders},
  author    = {Deng, Boyi and others},
  booktitle = {Proceedings of ACL},
  year      = {2025}
}
```