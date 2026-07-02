# Encoding vs. Linear Use of Patient Characteristics in CXR Foundation Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21142236.svg)](https://doi.org/10.5281/zenodo.21142236)

Analysis code and result tables for:

> **Encoding Versus Linear Use of Patient Characteristics in Chest X-Ray Foundation Models on MIMIC-CXR**
> Yeonsu Kim, Yangwon Kim, Yoojin Nam, Namjoon Kim, Pa Hong (corresponding author).
> *Diagnostics* **2026**, 16(13), 2030. https://doi.org/10.3390/diagnostics16132030

The paper separates three quantities on the same frozen embedding — how strongly a
patient attribute is *encoded*, how much finding prediction *depends* on it under
linear residualization, and the resulting *subgroup gap* — and shows the three
dissociate. This repository reproduces every number, table, and figure.

## What the code computes

For nine frozen chest X-ray foundation models on MIMIC-CXR (230,697 images,
60,518 patients):

1. **Encoding strength** of 24 patient attributes (4 demographics + 20 ICD
   comorbidities) with L2-regularized linear probes.
2. **Attribute dependence** — the AUROC drop when each attribute is residualized
   (ridge / Frisch–Waugh–Lovell) out of the frozen embedding before the finding
   probe is retrained.
3. **Attribute–finding odds ratios** from MIMIC-IV ICD codes.
4. **Nested OLS regression** of dependence on |log(OR)|, encoding strength, and
   model-level factors (effective rank, architecture), with cluster-robust and
   mixed-effects corrections.
5. **Cross-validation** (leave-one-finding-out, leave-one-attribute-out) and
   robustness checks (patient-level and inverse-probability-weighted ORs, view
   position, simultaneous residualization, U-Zero uncertainty labels, ICD timing).
6. **Fairness gap test** by sex and age tertiles, before vs. after residualizing
   the top-OR attributes, plus the four-category race subgroup analysis and
   nonlinear concept-erasure checks.

Given the fixed seeds in `experiments/config.py` and the same MIMIC-CXR v2.0 /
MIMIC-IV v3.1 cohorts, the results are deterministic up to GPU-level numerical
noise (see `REPRODUCIBILITY.md`).

## Repository layout

```
.
├── README.md
├── PIPELINE.md              step-by-step reproduction guide
├── REPRODUCIBILITY.md       data access, hardware, versions, CSV → table map
├── requirements.txt
├── LICENSE                  MIT
│
├── experiments/            analysis modules (run with PYTHONPATH=. )
│   ├── config.py             models, paths, seeds, attribute lists
│   ├── data.py               MIMIC loaders, split, CheXpert label merge
│   ├── embeddings.py         frozen-encoder embedding extraction helpers
│   ├── models.py             foundation-model wrappers
│   ├── stats.py              bootstrap CIs, DeLong test, FDR, permutation tests
│   ├── phase1_probing.py     encoding strength
│   ├── phase2_confounding.py residualization → dependence
│   ├── phase3_bias.py        subgroup fairness gaps
│   ├── phase4_fusion.py      integrated nested regression
│   ├── per_comorbidity_residualization*.py   24 × 10 × models dependence matrix
│   ├── race_*.py             race encoding, dependence, subgroup gaps
│   ├── two_factor_regression.py / nested_regression_n9.py   OR → dependence fits
│   ├── keystone_a/b_*.py     linearity and direct race-gap tests
│   ├── mcc_dependence.py, multimetric_fairness.py, nonlinear_concept_erasure.py
│   ├── *_or_sensitivity.py, robustness_exp*.py, icd_timing_sensitivity.py
│   └── chexpert_*.py         CheXpert Plus external checks
│
├── extract_full_mimic.py    MIMIC-CXR embedding extraction (all models)
├── extract_chexpert_plus.py CheXpert Plus embedding extraction
├── setup/                   MIMIC SQL schema + load scripts
├── figures/                 published figure PDFs + regeneration scripts
└── results/                 aggregate CSVs behind every table and figure
    results_chexpert/, results_chexpert_impression/   external-check outputs
```

The `results*/` CSVs are aggregate statistics only (per attribute / model / finding /
subgroup). No image, embedding, or patient-level record is included — those are
credentialed PhysioNet data and must be obtained separately (see below).

## Quick start

- `PIPELINE.md` — the full reproduction sequence, from embedding extraction to the
  final tables.
- `REPRODUCIBILITY.md` — how to obtain the data (PhysioNet credentialed access for
  MIMIC-CXR and MIMIC-IV), exact package versions, and which CSV backs which table.

The published numbers can be re-derived from the shipped `results/` CSVs without
re-running the GPU extraction; the analysis scripts read those CSVs directly.

## Data availability

MIMIC-CXR v2.0, MIMIC-CXR-JPG, and MIMIC-IV v3.1 are available through PhysioNet
under a credentialed data use agreement; CheXpert Plus is available from the Stanford
AIMI portal. None of these datasets is redistributed here. See `REPRODUCIBILITY.md`
for links and access requirements.

**Extracted embeddings and probe weights.** The paper's Data Availability Statement
refers to the extracted foundation-model embeddings. These are derived from
credentialed MIMIC-CXR and CheXpert data and cannot be redistributed here under the
PhysioNet and Stanford AIMI data use agreements. They can be regenerated exactly from
`extract_full_mimic.py` / `extract_chexpert_plus.py` using credentialed data and the
model weights listed in `REPRODUCIBILITY.md`, or obtained by credentialed users on
request to the corresponding author.

## Citation

```bibtex
@article{kim2026encoding,
  title   = {Encoding Versus Linear Use of Patient Characteristics in Chest
             X-Ray Foundation Models on MIMIC-CXR},
  author  = {Kim, Yeonsu and Kim, Yangwon and Nam, Yoojin and Kim, Namjoon and Hong, Pa},
  journal = {Diagnostics},
  volume  = {16},
  number  = {13},
  pages   = {2030},
  year    = {2026},
  doi     = {10.3390/diagnostics16132030}
}
```

To cite this code archive specifically (all versions), use the Zenodo concept DOI
[10.5281/zenodo.21142236](https://doi.org/10.5281/zenodo.21142236).

## Contact

Pa Hong (corresponding author) — papa.hong@samsung.com
Department of Radiology, Samsung Changwon Hospital,
Sungkyunkwan University School of Medicine.

## License

Code is released under the MIT License (see `LICENSE`). The datasets retain their
own licenses and data use agreements.
