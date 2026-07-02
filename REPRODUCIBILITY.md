# Reproducibility Guide

## Dataset access

All datasets are publicly available through PhysioNet under credentialed data use agreements. They are not redistributed in this repository.

| Dataset | Version | Access | Citation |
|---------|---------|--------|----------|
| MIMIC-CXR | v2.0 / v2.1.0 | https://physionet.org/content/mimic-cxr/ | Johnson et al., *Sci Data* 2019 |
| MIMIC-CXR-JPG | v2.0.0 | https://physionet.org/content/mimic-cxr-jpg/ | Johnson et al., 2019 |
| MIMIC-IV | v3.1 | https://physionet.org/content/mimiciv/ | Johnson et al., *Sci Data* 2023 |
| CheXpert Plus | v1.0 | https://stanfordaimi.azurewebsites.net | Chambon et al., 2024 |

To access PhysioNet data you must complete CITI Data or Specimens Only Research training and sign the project-specific DUA. Plan for ~1–2 weeks of approval lead time.

## Hardware

Reported wall-clock times correspond to:

- CPU: AMD Ryzen 9 9950X (16 cores, 32 threads)
- GPU: NVIDIA RTX 5090 (32 GB VRAM)
- RAM: 128 GB DDR5
- Storage: 8 TB NVMe SSD (for image cache + embeddings)
- OS: Windows 11 Pro / Linux Ubuntu 24.04

The pipeline runs on lower-end hardware (e.g., RTX 3090, 24 GB) with longer extraction time. CPU-only is feasible but extraction will take days.

## Software versions

Key package versions used to produce the published numbers:

| Package | Version |
|---------|---------|
| Python | 3.11.9 |
| PyTorch | 2.1.0 (cu121) |
| torchvision | 0.16.0 |
| scikit-learn | 1.4.0 |
| statsmodels | 0.14.1 |
| numpy | 1.26.3 |
| pandas | 2.1.4 |
| scipy | 1.11.4 |
| transformers | 4.36.2 |
| huggingface-hub | 0.20.2 |
| open-clip-torch | 2.24.0 |
| timm | 1.0.3 |
| torchxrayvision | 1.2.0 |

See [requirements.txt](requirements.txt) for the install spec.

## Configuring data paths

The credentialed data is not shipped with the code. Point the scripts at your local
copies either by editing the defaults in `experiments/config.py` or by setting these
environment variables:

| Variable | What it points to |
|----------|-------------------|
| `MIMIC_CXR_JPG_DIR` | MIMIC-CXR-JPG image root |
| `METADATA_CSV` | extracted per-image metadata table (see `extract_full_mimic.py`) |
| `CHEXPERT_CSV` | `mimic-cxr-2.0.0-chexpert.csv` finding labels |
| `EMBEDDING_DIR` | cached `.npy` embeddings written by the extraction step |
| `MIMIC_IV_ADMISSIONS` | MIMIC-IV `hosp/admissions.csv.gz` (the `race` field) |
| `CXR_MODEL_DIR` | local cache for downloaded model weights |

## Random seeds

All seeds are fixed in `experiments/config.py`:

- `SEED = 42` — sex-stratified patient-level 70/15/15 train/val/test split.
- MLP probe ensemble seeds: `0, 1, 2`.
- Bootstrap CIs: 10,000 resamples, deterministic given the data order.

## Foundation model weights

The 6 clean models are loaded from public sources:

| Model | Source | License |
|-------|--------|---------|
| ResNet50-ImageNet | torchvision (`ResNet50_Weights.IMAGENET1K_V2`) | BSD |
| DINOv2-base | facebookresearch/dinov2 | Apache 2.0 |
| BiomedCLIP | microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | MIT |
| XRV-DenseNet-nih | torchxrayvision `densenet121-res224-nih` | Apache 2.0 |
| CLIP-ViT-B/16 | openai/clip-vit-base-patch16 | MIT |
| ConvNeXtV2-Base | facebook/convnextv2-base-22k-224 | CC-BY-NC 4.0 (research only) |

Three overlap models used only in encoding/fairness analyses:

| Model | Source | Notes |
|-------|--------|-------|
| RAD-DINO | microsoft/rad-dino | MSRA license |
| CheXzero | rajpurkarlab/chexzero | MIT, MIMIC-CXR text-pretrained |
| CheSS | https://github.com/mi2rl/CheSS | Apache 2.0 |

## Mapping CSVs to supplementary tables

| Result CSV | Supplementary Table / Main Element |
|------------|---------------------|
| `results/phase1_probing.csv` | S1 (encoding column for 23 non-race attrs); S8 (per-finding baseline AUROCs) |
| `results/race_probing_mimic_cxr.csv` | S1 (race encoding row) |
| `results/per_comorbidity_residualization.csv` | S1 (24 × 6 dependence matrix; sex/age/BMI + 20 comorbidities) |
| `results/race_residualization_mimic_cxr.csv` | S1 (race dependence row, 6 clean models) |
| `results/per_comorbidity_residualization_overlap.csv` | encoding/fairness analyses on 3 CXR-overlap models (RAD-DINO, CheXzero, CheSS) |
| `results/race_residualization_overlap.csv` | S13 (race dependence, 3 overlap models incl. CheXzero) |
| `results/per_comorbidity_residualization_summary.csv` | Table 2 (waterfall summary) |
| `results/two_factor_regression_data.csv` | S4 (24 × 10 OR matrix); Figure 1B input — 23-attr legacy fit |
| `results/race_or_mimic_cxr.csv` | S4 (race × 10 findings OR row); Figure 1B race overlay |
| `results/two_factor_regression_data_with_race.csv`, `results/nested_regression_with_race.csv` | Table 3 race-inclusive primary fit (n = 1,440) |
| `results/two_factor_regression_data_n9.csv`, `results/nested_regression_n9.csv` | S12 (9-model sensitivity, 6 clean + 3 overlap) |
| `results/phase4_fusion.csv` | S6 (nested regression with cluster-robust + mixed-effects) |
| `results/patient_level_or_sensitivity.csv` | S7 (Patient-Level vs Image-Level OR) |
| `results/experiment1_ipw_or_comparison.csv` | S9 Panel A (IPW vs unweighted OR) |
| `results/experiment3_simultaneous_residualization.csv` | S9 Panel B (simultaneous residualization, redundancy = 0.440) |
| `results/biostat_diagnostics.csv` | S11 (OLS residual diagnostics + Huber sensitivity for M1) |
| `results/icd_timing_sensitivity.csv`, `results/icd_timing_sensitivity_full.csv` | S14 (adjudicable / non-adjudicable subset, β = 0.027) |
| `results/phase3_bias.csv`, `results/phase3_gaps.csv` | S3 (per-model fairness gaps) |
| `results/race_subgroup_gaps_mimic_cxr.csv`, `results/race_subgroup_summary_mimic_cxr.csv` | S13 (race subgroup gaps, 6 clean models) |
| `results/race_subgroup_summary_overlap.csv` | S13 (race subgroup gaps, 3 overlap models incl. CheXzero) |
| `results/fairness_debiasing_proof.csv` | Figure 3 (subgroup gap before/after top-3 residualization, MDES = 0.0019) |
| `results/mcc_dependence.csv` | S20 (dependence hierarchy under the Matthews correlation coefficient) |
| `results/multimetric_fairness_gaps.csv`, `results/multimetric_fairness_per_group.csv` | S21 (multi-metric subgroup fairness with bootstrap CIs) |
| `results/nonlinear_concept_erasure.csv`, `results/keystone_a_*.csv` | S18–S19 (nonlinear residualization / concept erasure) |
| `results/keystone_b_race_gap_*.csv`, `results/m4c_race_*.csv` | S15, S17 (direct race-gap test; per-group base rates and calibration) |
| `results/u_zero_or_sensitivity.csv` | U-Zero uncertainty-label OR sensitivity (Section 4.2) |
| `results_chexpert/*.csv`, `results_chexpert_impression/*.csv` | S16 (CheXpert Plus external check: encoding–dependence and race subgroup-gap dissociations) |
| `results/fairness_debiasing_comparison.csv` | sensitivity artifact (top-3 vs all-confounder debiasing); not cited as a supplementary table |
| `results/experiment2_view_position_or.csv` | sensitivity artifact (AP/PA stratified OR); superseded by the Mantel-Haenszel check reported in Section 4.2 |
| `results/supp_nonlinear_residualization.csv` | sensitivity artifact (linear vs MLP residualization); superseded by S18 |

Pairwise rank correlation values for **Table S10** are computed inline from `per_comorbidity_residualization.csv` + `race_residualization_mimic_cxr.csv` by `experiments/two_factor_regression.py`.

LOFO/LOAO cross-validation outputs (**Table S2**) are computed inline by `experiments/two_factor_regression.py` and reported to stdout.

Cohort descriptive statistics and label distributions are computed by inline summaries in `experiments/data.py`.

LoRA fine-tuning numbers (**Table S5**) are reported as stand-alone values in the supplementary and are not regenerated by this code release. As the supplementary note states, post-LoRA metrics are estimates from the original RAD-DINO fine-tuning experiment; the LoRA generation script is not included in this code release.

## Known sources of non-determinism

The pipeline is fully deterministic *given* fixed seeds **except** for:

1. **CUDA non-determinism** in the embedding extraction step. Bit-exact reproducibility of embeddings across different GPU models is not guaranteed (e.g., RTX 3090 vs. RTX 5090). Downstream AUROC numbers may differ at the 4th–5th decimal. The qualitative findings (rankings, sign of effects, $p$-value classes) are stable.
2. **MIMIC-CXR JPEG decoding**: Pillow versions before 10.0 used a different JPEG decoder. We pin Pillow >= 10.0 in `requirements.txt`.

Differences larger than 0.001 in AUROC or 0.01 in $R^2$ should be reported as a reproducibility issue (open a GitHub issue).

## Contact for reproduction questions

Pa Hong, papa.hong@samsung.com (corresponding author).
