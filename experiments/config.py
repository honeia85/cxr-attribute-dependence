"""
Central configuration for all experiments.
All constants, paths, hyperparameters in one place.
"""
import os

# --- Paths ---
# Data locations default to sitting alongside this repository. If your copies of
# the (credentialed) MIMIC data live elsewhere, either edit the defaults below or
# set the corresponding environment variable.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.environ.get("MIMIC_CXR_JPG_DIR", os.path.join(PROJECT_DIR, "mimic-cxr-jpg"))
EMBEDDING_DIR = os.environ.get("EMBEDDING_DIR", os.path.join(PROJECT_DIR, "embeddings_full"))
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
FIGURE_DIR = os.path.join(PROJECT_DIR, "figures")
MODEL_DIR = os.environ.get("CXR_MODEL_DIR", os.path.join(PROJECT_DIR, "cxr_foundation_models"))

# Source tables extracted from MIMIC-CXR / MIMIC-IV (see extract_full_mimic.py and setup/).
# These hold credentialed PhysioNet data and are not shipped with the code.
METADATA_CSV = os.environ.get("METADATA_CSV", os.path.join(PROJECT_DIR, "full_mimic_cxr_metadata.csv"))
CHEXPERT_CSV = os.environ.get("CHEXPERT_CSV", os.path.join(PROJECT_DIR, "mimic-cxr-2.0.0-chexpert.csv"))

# MIMIC-IV hosp/admissions.csv.gz, used for the admissions.race field in the race analyses.
MIMIC_IV_ADMISSIONS = os.environ.get(
    "MIMIC_IV_ADMISSIONS", os.path.join(PROJECT_DIR, "mimic-iv", "hosp", "admissions.csv.gz")
)

# Canonical embedding-order artifacts (written by the extraction step).
CANONICAL_IDS = os.path.join(EMBEDDING_DIR, "canonical_dicom_ids.npy")
CANONICAL_SPLIT = os.path.join(EMBEDDING_DIR, "canonical_split.npz")

# Site-specific ID column and file suffix
ID_COL = "dicom_id"
IDS_FILE_SUFFIX = "dicom_ids"

# ============================================================
# Random Seeds
# ============================================================
SEED = 42
MULTI_SEEDS = [42, 123, 456, 789, 1024]

# ============================================================
# Foundation Models
# ============================================================
MODELS = {
    "ResNet50-ImageNet": {
        "category": "General-purpose supervised",
        "embed_dim": 2048,
        "framework": "pytorch",
        "input_size": 224,
    },
    "DINOv2-base": {
        "category": "General-purpose self-supervised",
        "embed_dim": 768,
        "framework": "pytorch",
        "input_size": 518,
        "hf_name": "facebook/dinov2-base",
    },
    "XRV-DenseNet-nih": {
        "category": "CXR label-supervised",
        "embed_dim": 1024,
        "framework": "pytorch",
        "input_size": 224,
    },
    "RAD-DINO": {
        "category": "CXR self-supervised",
        "embed_dim": 768,
        "framework": "pytorch",
        "input_size": 518,
        "hf_name": "microsoft/rad-dino",
    },
    "BiomedCLIP": {
        "category": "General-purpose VLM",
        "embed_dim": 512,
        "framework": "pytorch",
        "input_size": 224,
        "hf_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    },
    "CheXzero": {
        "category": "CXR report-contrastive",
        "embed_dim": 512,
        "framework": "pytorch",
        "input_size": 224,
    },
    "CheSS-ResNet50": {
        "category": "CXR self-supervised",
        "embed_dim": 2048,
        "framework": "pytorch",
        "input_size": 224,
    },
    "CLIP-ViT-B16": {
        "category": "General-purpose VLM",
        "embed_dim": 512,
        "framework": "pytorch",
        "input_size": 224,
        "hf_name": "openai/clip-vit-base-patch16",
    },
    "ConvNeXtV2-Base": {
        "category": "General-purpose supervised",
        "embed_dim": 1024,
        "framework": "pytorch",
        "input_size": 224,
        "hf_name": "facebook/convnextv2-base-22k-224",
    },
}

# Strict clean models (no pretraining-evaluation data overlap) — primary analysis
CLEAN_MODELS = [
    "ResNet50-ImageNet", "DINOv2-base", "XRV-DenseNet-nih", "BiomedCLIP",
    "CLIP-ViT-B16", "ConvNeXtV2-Base",
]

# Uncertain overlap (pretraining data partially opaque) — sensitivity analysis
UNCERTAIN_MODELS = ["CheSS-ResNet50"]

# Confirmed overlap (pretrained on MIMIC/CheXpert) — supplementary analysis
OVERLAP_MODELS = ["RAD-DINO", "CheXzero"]

# Backward-compatible alias: primary analysis uses strict clean models
MAIN_MODELS = CLEAN_MODELS

# Supplementary models (XRV variants)
XRV_VARIANTS = ["all", "chex", "nih"]

# Short display names for plots
MODEL_SHORT_NAMES = {
    "ResNet50-ImageNet": "ResNet50-IN",
    "DINOv2-base": "DINOv2",
    "XRV-DenseNet-nih": "XRV",
    "RAD-DINO": "RAD-DINO",
    "BiomedCLIP": "BiomedCLIP",
    "CheXzero": "CheXzero",
    "CheSS-ResNet50": "CheSS",
    "CLIP-ViT-B16": "CLIP",
    "ConvNeXtV2-Base": "ConvNeXtV2",
}

# Model colors for consistent visualization
MODEL_COLORS = {
    "ResNet50-ImageNet": "#95a5a6",
    "DINOv2-base": "#f39c12",
    "XRV-DenseNet-nih": "#3498db",
    "RAD-DINO": "#2ecc71",
    "BiomedCLIP": "#e74c3c",
    "CheXzero": "#9b59b6",
    "CheSS-ResNet50": "#1abc9c",
    "CLIP-ViT-B16": "#e67e22",
    "ConvNeXtV2-Base": "#2c3e50",
}

# ============================================================
# CheXpert Disease Labels
# ============================================================
CHEXPERT_ALL = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pneumonia", "Pneumothorax", "Support Devices",
]

# Excluded from primary analysis (too few labels / below-random performance)
CHEXPERT_EXCLUDE = ["Fracture", "Lung Lesion"]

# Minimum thresholds for disease inclusion in analyses
MIN_VALID_LABELS = 100  # minimum non-masked labels in test set
MIN_POSITIVE = 30       # minimum positive cases in test set
MIN_NEGATIVE = 30       # minimum negative cases in test set
MAX_PREVALENCE = 0.90   # exclude diseases with >90% prevalence (degenerate classification)

# ============================================================
# Probing Targets (Phase 1)
# ============================================================
BINARY_TARGETS = [
    "gender_binary", "diabetes", "hypertension", "ckd", "heart_failure",
    "copd", "smoking_history", "cancer_history", "atrial_fibrillation",
    "coronary_artery_disease", "obesity", "anemia", "depression",
    "hypothyroidism", "respiratory_failure", "aki", "stroke",
    "pulmonary_fibrosis", "liver_disease", "hyperlipidemia", "asthma",
]

REGRESSION_TARGETS = ["age", "bmi", "sbp", "dbp"]

# Minimum samples for probing
PROBE_MIN_POS = 50
PROBE_MIN_NEG = 50
PROBE_MIN_REGRESSION = 200

# ============================================================
# Metadata Features (for Fusion / Confounding)
# ============================================================
CONTINUOUS_FEATURES = ["age", "bmi", "sbp", "dbp"]
BINARY_FEATURES = [
    "gender_binary", "hypertension", "heart_failure", "atrial_fibrillation",
    "coronary_artery_disease", "stroke", "diabetes", "hyperlipidemia",
    "obesity", "hypothyroidism", "ckd", "aki", "copd", "asthma",
    "respiratory_failure", "pulmonary_fibrosis", "liver_disease",
    "anemia", "smoking_history", "cancer_history", "depression",
]
NUM_META_FEATURES = len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES)

# Confounders for Phase 2
CONFOUNDERS = ["age", "gender_binary", "bmi"]

# ============================================================
# Subgroup Definitions (Phase 3)
# ============================================================
SUBGROUPS = {
    "Gender": {
        "column": "gender_binary",
        "bins": None,  # binary: 0=Female, 1=Male
        "labels": ["Female", "Male"],
    },
    "Age": {
        "column": "age",
        "bins": [0, 50, 70, 200],
        "labels": ["<50", "50-70", "70+"],
    },
    "BMI": {
        "column": "bmi",
        "bins": [0, 25, 30, 200],
        "labels": ["<25", "25-30", "30+"],
    },
}

# Minimum samples per subgroup cell for bias analysis
BIAS_MIN_VALID = 50
BIAS_MIN_POS = 15
BIAS_MIN_NEG = 15

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6
GRAD_CLIP = 1.0

# ============================================================
# Statistical Testing
# ============================================================
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95
FDR_Q = 0.05       # Benjamini-Hochberg FDR threshold
N_PERMUTATIONS = 1000  # for permutation null baseline

# ============================================================
# Adversarial Debiasing (Phase 2 Method B)
# ============================================================
ADV_ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]  # gentler range to avoid signal destruction
ADV_TARGET_AUROC = 0.60  # adversary should drop below this (relaxed from 0.55)

# ============================================================
# Phase 2 Classification Thresholds (Method A primary)
# ============================================================
CONF_GENUINE_MAX_DROP = 0.03    # Method A drop < 3% AND added > 5%
CONF_GENUINE_MIN_ADDED = 0.05
CONF_HEAVY_MIN_DROP = 0.08     # Method A drop >= 8%
CONF_HEAVY_MAX_ADDED = 0.03   # OR added value < 3%
