# Reproduction Pipeline

End-to-end sequence to reproduce all numbers in the manuscript.
Estimated wall-clock on a single NVIDIA RTX 5090: ~24 hours
(dominated by embedding extraction; analysis stages take minutes each).

## 0. Prerequisites

- Python 3.11+ with packages from [requirements.txt](requirements.txt).
- PhysioNet credentialed access (CITI training certificate) for MIMIC-CXR v2.0 and MIMIC-IV v3.1. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- ~250 GB free disk for raw JPEGs + ~15 GB for cached embeddings.
- A CUDA-capable GPU is recommended (CPU-only is feasible but slow for ViT models).

Point the code at your local data by editing `experiments/config.py` (or by setting
the matching environment variables): `IMAGE_DIR` / `MIMIC_CXR_JPG_DIR`, `METADATA_CSV`,
`EMBEDDING_DIR`, and `MIMIC_IV_ADMISSIONS` (the `hosp/admissions.csv.gz` file from
MIMIC-IV, used for the `race` field in the race analyses).

## 1. Data preparation

```bash
psql -f setup/create_mimic4.sql
psql -f setup/load_mimic4.sql
```

Then run the SQL queries in `setup/` to export `patient_metadata.csv` (subject_id × 20 binary comorbidities + age, sex, BMI).

## 2. Embedding extraction (~12-18 hours on RTX 5090)

```bash
python extract_full_mimic.py --model all --batch-size 32
python extract_chexpert_plus.py --model all --batch-size 32   # optional, race sensitivity
```

Output: `embeddings_full/<model_name>.npy` (one file per model, ~700 MB to 1.8 GB each).

## 3. Encoding strength (Phase 1)

```bash
python -m experiments.phase1_probing
```

Output: `results/phase1_probing.csv` → cited as Table S1 (encoding column) and Table S8 (per-finding baseline AUROCs).

## 4. Attribute dependence via residualization (Phase 2)

```bash
python -m experiments.per_comorbidity_residualization
```

Output: `results/per_comorbidity_residualization.csv` → 24 attributes × 10 findings × 6 clean models = 1,440 dependence cells (Table 1, Table S1, Figure 1A, Figure 2).

For the three CXR-overlap models (RAD-DINO, CheXzero, CheSS) used only in encoding/fairness analyses:

```bash
python -m experiments.per_comorbidity_residualization_overlap
```

Output: `results/per_comorbidity_residualization_overlap.csv`.

Race (the 4th demographic) is computed by dedicated scripts because four-category race requires a one-hot residualization block:

```bash
python -m experiments.race_probing_mimic_cxr           # race encoding AUROC (Table 1 row)
python -m experiments.race_residualization_mimic_cxr   # race dependence on clean models
python -m experiments.race_residualization_overlap     # race dependence on overlap models
```

Outputs: `results/race_probing_mimic_cxr.csv`, `results/race_residualization_mimic_cxr.csv`, `results/race_residualization_overlap.csv`, `results/race_subgroup_gaps_mimic_cxr.csv`, `results/race_subgroup_summary_mimic_cxr.csv`, `results/race_subgroup_summary_overlap.csv`, `results/race_or_mimic_cxr.csv`.

## 5. Odds ratios

```bash
python -m experiments.patient_level_or_sensitivity
python -m experiments.robustness_exp1_ipw_or
```

Outputs:
- `results/two_factor_regression_data.csv` — image-level OR table (Table S4 input, 23-attr legacy).
- `results/patient_level_or_sensitivity.csv` — Table S7 (Patient-Level vs Image-Level OR).
- `results/experiment1_ipw_or_comparison.csv` — IPW OR robustness (Table S9 Panel A).

## 6. Nested OLS regression (Table 3)

```bash
python -m experiments.two_factor_regression
python -m experiments.integrate_race_into_regression   # adds race as 24th attribute
python -m experiments.nested_regression_n9             # 9-model sensitivity (6 clean + 3 overlap)
```

Outputs:
- `results/phase4_fusion.csv` → models M1, M1a, M2, M5, M6a, M6b, M7a, M7b, M8 (Table 3, Table S6 with cluster-robust + mixed-effects corrections).
- `results/two_factor_regression_data_with_race.csv`, `results/nested_regression_with_race.csv` → race-inclusive regression input and fit (n = 1,440 primary).
- `results/two_factor_regression_data_n9.csv`, `results/nested_regression_n9.csv` → 9-model robustness fit (Table S12).

## 7. Cross-validation and robustness

```bash
python -m experiments.encoding_confounding_robustness    # Linear vs MLP probes (sensitivity artifact, not in v5 supp)
python -m experiments.robustness_exp2_viewposition       # AP/PA stratified (sensitivity artifact, not in v5 supp)
python -m experiments.robustness_exp3_simultaneous       # Simultaneous residualization (Table S9 Panel B)
python -m experiments.icd_timing_sensitivity             # non-adjudicable subset (Table S14)
python -m experiments.biostat_diagnostics                # OLS residual diagnostics (Table S11)
```

Outputs (in addition to those above):
- `results/icd_timing_sensitivity.csv`, `results/icd_timing_sensitivity_full.csv` → Table S14 (ICD timing / label circularity bound).
- `results/biostat_diagnostics.csv` → Table S11 (OLS residual diagnostics + Huber sensitivity for M1).
- `results/experiment3_simultaneous_residualization.csv` → Table S9 Panel B (simultaneous, redundancy = 0.440).

## 8. Fairness gap test (Figure 3)

```bash
python -m experiments.phase3_bias              # Per-subgroup AUROC gaps (Figure 3 raw)
python -m experiments.fairness_debiasing_proof # Before vs. after top-3 residualization, MDES = 0.0019
```

Outputs: `results/phase3_bias.csv`, `results/phase3_gaps.csv`, `results/fairness_debiasing_proof.csv`.

## 9. Supplementary and revision analyses

These scripts produce the remaining supplementary tables (S13–S21) and the
robustness checks. Each reads the cached embeddings / result CSVs and writes to
`results/` (or `results_chexpert*/`):

```bash
python -m experiments.mcc_dependence                    # S20: MCC-based dependence
python -m experiments.multimetric_fairness              # S21: multi-metric subgroup fairness
python -m experiments.nonlinear_concept_erasure         # S19: kernel-space concept erasure
python -m experiments.keystone_a_nonlinear_demographic  # S18: linear vs nonlinear use
python -m experiments.keystone_b_race_comorbidity_residualization  # S15: direct race-gap test
python -m experiments.m4c_race_baserate_calibration     # S17: race base rates + calibration
python -m experiments.u_zero_or_sensitivity             # U-Zero uncertainty-label OR check
python -m experiments.chexpert_external_validation      # S16: CheXpert Plus external check
```

## 10. Figures

`figures/` contains the published figure PDFs and the scripts that regenerate them
from the result CSVs:

```bash
python figures/figure1.py   # Figure 1: encoding–dependence + OR–dependence scatter
python figures/figure3.py   # Figure 3: subgroup-gap fairness null
```

Figure 2 (the dependence waterfall) is a direct bar chart of
`results/per_comorbidity_residualization_summary.csv`. The figure scripts are not
required to verify the numbers — the CSVs are sufficient.

## Verifying against the manuscript

Every numerical claim in the manuscript and supplementary tables can be cross-checked against the result CSVs. For example:

- Figure 1A "sex AUROC 0.942" → `results/phase1_probing.csv`, attribute = `sex`.
- Table 3 "heart failure +0.018 [+0.013, +0.023]" → `results/per_comorbidity_residualization_summary.csv`.
- Table 4 M1 "$R^2$ = 0.506" → `results/phase4_fusion.csv`, model = M1.
- Figure 3 "MDES = 0.0019" → `results/fairness_debiasing_proof.csv`.

The mapping of CSVs to supplementary tables is documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
