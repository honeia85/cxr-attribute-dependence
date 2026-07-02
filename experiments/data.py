"""
Canonical data preparation: loading, splitting, preprocessing.
All experiments share the same data pipeline through this module.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from experiments.config import (
    PROJECT_DIR, IMAGE_DIR, EMBEDDING_DIR,
    METADATA_CSV, CHEXPERT_CSV, CANONICAL_IDS, CANONICAL_SPLIT,
    ID_COL, SEED, CHEXPERT_ALL, CHEXPERT_EXCLUDE,
    MIN_VALID_LABELS, MIN_POSITIVE, MIN_NEGATIVE, MAX_PREVALENCE,
    CONTINUOUS_FEATURES, BINARY_FEATURES,
    BINARY_TARGETS, REGRESSION_TARGETS,
    PROBE_MIN_POS, PROBE_MIN_NEG, PROBE_MIN_REGRESSION,
    MODELS,
)


def load_metadata():
    """Load sample_cxr_metadata.csv and add derived columns."""
    df = pd.read_csv(METADATA_CSV)
    if "gender_binary" not in df.columns:
        df["gender_binary"] = (df["gender"] == "M").astype(int)
    return df


def load_chexpert():
    """Load CheXpert labels."""
    return pd.read_csv(CHEXPERT_CSV)


def get_valid_image_paths(df):
    """Filter to rows where image file exists on disk."""
    mask = df["img_path"].apply(
        lambda p: os.path.exists(os.path.join(IMAGE_DIR, str(p)))
    )
    return df[mask].reset_index(drop=True)


def build_canonical_dataset(embedding_dict=None, models=None):
    """Build the canonical dataset: intersection of all models' valid images.

    Args:
        embedding_dict: dict of {model_name: (embeddings, dicom_ids)} or None.
            If None, loads from cached files in EMBEDDING_DIR.
        models: list of model names to include. If None, uses all in MODELS.

    Returns:
        canonical_ids: np.array of dicom_ids in canonical order
        canonical_df: DataFrame aligned to canonical_ids
    """
    df = load_metadata()
    df = get_valid_image_paths(df)
    valid_ids = set(df["dicom_id"].values)
    print(f"  Valid images in metadata: {len(valid_ids)}")

    model_list = models if models is not None else list(MODELS.keys())

    # Intersect with all model embeddings
    if embedding_dict is None:
        embedding_dict = {}
        for model_name in model_list:
            emb_path = os.path.join(EMBEDDING_DIR, f"{model_name}_embeddings.npy")
            ids_path = os.path.join(EMBEDDING_DIR, f"{model_name}_dicom_ids.npy")
            if os.path.exists(emb_path) and os.path.exists(ids_path):
                ids = np.load(ids_path, allow_pickle=True)
                embedding_dict[model_name] = ids

    for model_name, ids in embedding_dict.items():
        if isinstance(ids, tuple):
            ids = ids[1]  # (embeddings, dicom_ids)
        model_ids = set(ids)
        valid_ids = valid_ids & model_ids
        print(f"  After {model_name}: {len(valid_ids)} images")

    # Sort canonical IDs for reproducibility
    canonical_ids = np.array(sorted(valid_ids))
    print(f"  Canonical dataset: {len(canonical_ids)} images")

    # Align DataFrame
    canonical_df = df[df["dicom_id"].isin(set(canonical_ids))].copy()
    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    canonical_df["_canon_idx"] = canonical_df["dicom_id"].map(id_to_order)
    canonical_df = canonical_df.sort_values("_canon_idx").reset_index(drop=True)

    # Save
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    np.save(CANONICAL_IDS, canonical_ids)
    print(f"  Saved: {CANONICAL_IDS}")

    return canonical_ids, canonical_df


def create_split(canonical_df):
    """Create train/val/test split at the PATIENT level, stratified by sex.

    Patient-level splitting prevents data leakage when multiple images
    belong to the same patient (common in MIMIC-CXR: ~3.8 images/patient).

    Returns:
        dict with 'train_idx', 'val_idx', 'test_idx' arrays (image-level indices)
    """
    # Get unique patients and their sex
    patient_col = "subject_id" if "subject_id" in canonical_df.columns else "dicom_id"
    patient_df = canonical_df.groupby(patient_col).agg(
        gender_binary=("gender_binary", "first")
    ).reset_index()

    patients = patient_df[patient_col].values
    patient_gender = patient_df["gender_binary"].values

    # Split patients 70/15/15
    train_patients, temp_patients = train_test_split(
        patients, test_size=0.3, random_state=SEED, stratify=patient_gender
    )
    temp_mask = patient_df[patient_col].isin(temp_patients)
    temp_gender = patient_df[temp_mask]["gender_binary"].values
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=SEED, stratify=temp_gender
    )

    # Map patient-level split to image-level indices
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    train_idx = np.where(canonical_df[patient_col].isin(train_set))[0]
    val_idx = np.where(canonical_df[patient_col].isin(val_set))[0]
    test_idx = np.where(canonical_df[patient_col].isin(test_set))[0]

    split = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
    np.savez(CANONICAL_SPLIT, **split)
    n_patients = len(patients)
    print(f"  Patients: {n_patients} -> train={len(train_patients)}, "
          f"val={len(val_patients)}, test={len(test_patients)}")
    print(f"  Images:  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Saved: {CANONICAL_SPLIT}")
    return split


def load_split():
    """Load cached split."""
    data = np.load(CANONICAL_SPLIT)
    return {k: data[k] for k in data.files}


def load_canonical_ids():
    """Load cached canonical dicom IDs."""
    return np.load(CANONICAL_IDS, allow_pickle=True)


def get_aligned_embeddings(model_name, canonical_ids):
    """Load embeddings for a model, aligned to canonical order.

    Returns:
        np.array of shape (n_canonical, embed_dim)
    """
    emb_path = os.path.join(EMBEDDING_DIR, f"{model_name}_embeddings.npy")
    ids_path = os.path.join(EMBEDDING_DIR, f"{model_name}_dicom_ids.npy")

    embeddings = np.load(emb_path)
    dicom_ids = np.load(ids_path, allow_pickle=True)

    id_to_idx = {did: i for i, did in enumerate(dicom_ids)}
    aligned_idx = [id_to_idx[did] for did in canonical_ids]
    return embeddings[aligned_idx]


def merge_chexpert(canonical_df):
    """Merge canonical DataFrame with CheXpert labels.

    Returns:
        merged_df: DataFrame with CheXpert columns added
        labels: np.array (n, n_diseases) float32
        masks: np.array (n, n_diseases) bool - True where label is valid
    """
    diseases = [d for d in CHEXPERT_ALL if d not in CHEXPERT_EXCLUDE]

    # Skip merge if CheXpert columns already present (e.g. full_mimic_cxr_metadata.csv)
    if all(d in canonical_df.columns for d in diseases):
        merged = canonical_df.copy()
    else:
        chexpert = load_chexpert()
        merged = canonical_df.merge(
            chexpert, on=["subject_id", "study_id"], how="inner"
        )

    # Keep canonical order
    id_to_order = {did: i for i, did in enumerate(canonical_df[ID_COL].values)}
    merged["_order"] = merged[ID_COL].map(id_to_order)
    merged = merged.sort_values("_order").reset_index(drop=True)

    # Process labels: U-Ignore strategy
    diseases = [d for d in CHEXPERT_ALL if d not in CHEXPERT_EXCLUDE]
    labels = merged[diseases].values.copy()
    masks = np.zeros_like(labels, dtype=bool)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            val = labels[i, j]
            if val == 1.0:
                masks[i, j] = True
            elif val == 0.0:
                masks[i, j] = True
            else:
                labels[i, j] = 0.0  # mask out

    return merged, labels.astype(np.float32), masks, diseases


def get_eligible_diseases(labels, masks, split, diseases):
    """Determine which diseases meet inclusion criteria for primary analysis.

    Returns:
        tier1: list of disease names meeting full criteria
        tier2: list of disease names meeting relaxed criteria
    """
    test_idx = split["test_idx"]
    test_labels = labels[test_idx]
    test_masks = masks[test_idx]

    tier1, tier2 = [], []
    excluded_prev = []
    for j, disease in enumerate(diseases):
        valid = test_masks[:, j]
        n_valid = valid.sum()
        n_pos = test_labels[valid, j].sum() if n_valid > 0 else 0
        n_neg = n_valid - n_pos
        prevalence = n_pos / n_valid if n_valid > 0 else 1.0

        # Skip degenerate diseases (extreme imbalance)
        if prevalence > MAX_PREVALENCE or (1 - prevalence) > MAX_PREVALENCE:
            excluded_prev.append((disease, prevalence))
            continue

        if n_valid >= MIN_VALID_LABELS and n_pos >= MIN_POSITIVE and n_neg >= MIN_NEGATIVE:
            tier1.append(disease)
        elif n_valid >= 30 and n_pos >= 15 and n_neg >= 15:
            tier2.append(disease)

    if excluded_prev:
        print(f"  Excluded {len(excluded_prev)} diseases (prevalence > {MAX_PREVALENCE:.0%}):")
        for name, prev in excluded_prev:
            print(f"    {name}: {prev:.1%}")

    return tier1, tier2


def build_metadata_vectors(df, split):
    """Build metadata feature vectors with proper train-fit scaling.

    Returns:
        train_meta, val_meta, test_meta: np.arrays of shape (n, NUM_META_FEATURES)
        meta_scaler: fitted StandardScaler (for continuous features)
    """
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    # Continuous: fit scaler on train, impute with train median
    avail_cont = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    cont = df[avail_cont].copy()
    train_medians = cont.iloc[train_idx].median()
    for col in avail_cont:
        cont[col] = cont[col].fillna(train_medians[col])

    meta_scaler = StandardScaler()
    cont_train = meta_scaler.fit_transform(cont.iloc[train_idx].values)
    cont_val = meta_scaler.transform(cont.iloc[val_idx].values)
    cont_test = meta_scaler.transform(cont.iloc[test_idx].values)

    # Binary: fillna(0)
    avail_bin = [c for c in BINARY_FEATURES if c in df.columns]
    binary = df[avail_bin].fillna(0).values.astype(np.float32)
    bin_train = binary[train_idx]
    bin_val = binary[val_idx]
    bin_test = binary[test_idx]

    train_meta = np.concatenate([cont_train, bin_train], axis=1).astype(np.float32)
    val_meta = np.concatenate([cont_val, bin_val], axis=1).astype(np.float32)
    test_meta = np.concatenate([cont_test, bin_test], axis=1).astype(np.float32)

    return train_meta, val_meta, test_meta, meta_scaler


def scale_embeddings(embeddings, split):
    """StandardScale embeddings: fit on train, transform all.

    Returns:
        train_emb, val_emb, test_emb: scaled np.arrays
        emb_scaler: fitted StandardScaler
    """
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    emb_scaler = StandardScaler()
    train_emb = emb_scaler.fit_transform(embeddings[train_idx])
    val_emb = emb_scaler.transform(embeddings[val_idx])
    test_emb = emb_scaler.transform(embeddings[test_idx])

    return train_emb, val_emb, test_emb, emb_scaler


def get_probing_targets(df):
    """Get valid probing targets and their data.

    Returns:
        targets: list of dicts with 'name', 'task', 'y', 'mask' keys
    """
    targets = []

    for name in BINARY_TARGETS:
        if name not in df.columns:
            continue
        y = df[name].values.astype(float)
        mask = ~np.isnan(y)
        n_pos = y[mask].sum()
        n_neg = mask.sum() - n_pos
        if n_pos >= PROBE_MIN_POS and n_neg >= PROBE_MIN_NEG:
            targets.append({
                "name": name, "task": "binary",
                "y": y, "mask": mask,
                "prevalence": n_pos / mask.sum(),
            })

    for name in REGRESSION_TARGETS:
        if name not in df.columns:
            continue
        y = df[name].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() >= PROBE_MIN_REGRESSION:
            targets.append({
                "name": name, "task": "regression",
                "y": y, "mask": mask,
            })

    return targets


def print_dataset_summary(canonical_df, split, labels, masks, diseases):
    """Print summary statistics for the canonical dataset."""
    print("\n" + "=" * 60)
    print("  CANONICAL DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total samples: {len(canonical_df)}")
    print(f"  Train: {len(split['train_idx'])}, Val: {len(split['val_idx'])}, "
          f"Test: {len(split['test_idx'])}")

    # Demographics
    print(f"\n  Demographics:")
    print(f"    Gender (male %): {canonical_df['gender_binary'].mean():.1%}")
    print(f"    Age: {canonical_df['age'].mean():.1f} +/- {canonical_df['age'].std():.1f}")
    bmi = canonical_df["bmi"].dropna()
    print(f"    BMI: {bmi.mean():.1f} +/- {bmi.std():.1f} (n={len(bmi)})")

    # Disease label coverage
    print(f"\n  CheXpert Label Coverage (test set):")
    test_idx = split["test_idx"]
    for j, disease in enumerate(diseases):
        valid = masks[test_idx, j]
        n_valid = valid.sum()
        n_pos = labels[test_idx][valid, j].sum()
        print(f"    {disease:30s}: {n_valid:4d} valid, "
              f"{int(n_pos):4d} pos ({n_pos/n_valid:.1%})" if n_valid > 0 else
              f"    {disease:30s}: 0 valid")
