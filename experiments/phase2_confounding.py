"""
Phase 2: Confounding Analysis - Are disease predictions using demographic shortcuts?

Three complementary methods:
  A. Linear Residualization (embedding-space debiasing)
  B. Adversarial Debiasing (gradient reversal, gentler alphas)
  C. Confounder-Only Baseline (how much can demographics alone predict?)

Classification based primarily on Method A (stable, interpretable).
Adversarial results reported as secondary evidence.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

from experiments.config import (
    MODELS, MAIN_MODELS, SEED, MULTI_SEEDS, RESULT_DIR, CONFOUNDERS,
    ADV_ALPHAS, ADV_TARGET_AUROC, BATCH_SIZE, EPOCHS, LR, PATIENCE,
    CONF_GENUINE_MAX_DROP, CONF_GENUINE_MIN_ADDED,
    CONF_HEAVY_MIN_DROP, CONF_HEAVY_MAX_ADDED, ID_COL,
)
from experiments.data import (
    load_metadata, load_canonical_ids, load_split,
    get_aligned_embeddings, merge_chexpert, get_eligible_diseases,
    build_metadata_vectors, scale_embeddings,
)
from experiments.stats import bootstrap_delta, format_delta
from experiments.models import (
    set_all_seeds, CXROnlyModel, AdversarialModel,
    compute_pos_weight, masked_bce_loss,
    train_disease_model, predict_disease, compute_per_disease_auroc,
    make_loaders, EmbeddingDataset,
)


def method_a_residualization(embeddings, confounder_values, split):
    """Method A: Remove linear projection of confounders from embeddings.

    Args:
        embeddings: (n, embed_dim) raw embeddings
        confounder_values: (n, n_confounders) confounder array
        split: train/val/test indices

    Returns:
        residualized embeddings (n, embed_dim), r2_explained (float)
    """
    train_idx = split["train_idx"]
    Z = confounder_values.copy()

    # Impute NaN with train median
    for j in range(Z.shape[1]):
        median_val = np.nanmedian(Z[train_idx, j])
        Z[np.isnan(Z[:, j]), j] = median_val

    # Fit OLS on train: embedding_dim_i = Z @ beta + residual
    reg = LinearRegression()
    reg.fit(Z[train_idx], embeddings[train_idx])
    predicted = reg.predict(Z)
    residuals = embeddings - predicted

    # Compute R2: how much of embedding variance is explained by confounders
    r2_train = r2_score(embeddings[train_idx], predicted[train_idx],
                        multioutput="variance_weighted")

    # Verify: confounders should be unpredictable from residuals
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    res_train = sc.fit_transform(residuals[train_idx])
    res_test = sc.transform(residuals[split["test_idx"]])

    gender_idx = CONFOUNDERS.index("gender_binary")
    gender_train = Z[train_idx, gender_idx]
    gender_test = Z[split["test_idx"], gender_idx]
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(res_train, gender_train)
    gender_pred = clf.predict_proba(res_test)[:, 1]
    gender_auroc = roc_auc_score(gender_test, gender_pred)
    print(f"    Residual -> Gender AUROC: {gender_auroc:.3f} (should be ~0.5)")
    print(f"    Confounder R2 on embeddings: {r2_train:.4f}")

    return residuals, r2_train


def method_b_adversarial(train_emb, val_emb, test_emb,
                         train_labels, val_labels, test_labels,
                         train_masks, val_masks, test_masks,
                         train_conf, val_conf, test_conf,
                         cxr_dim, num_labels, disease_names,
                         pos_weight=None, seed=SEED):
    """Method B: Adversarial debiasing with gradient reversal.

    Uses gentler alpha range to avoid destroying disease signal.

    Returns:
        dict of results per alpha, best_result (at target), best_alpha
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_alpha_results = {}
    best_result = None
    best_alpha = None

    for alpha in ADV_ALPHAS:
        set_all_seeds(seed)
        model = AdversarialModel(cxr_dim, num_labels, num_confounders=len(CONFOUNDERS))
        model.adv_alpha = alpha
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        train_ds = EmbeddingDataset(train_emb, train_labels, train_masks, train_conf)
        val_ds = EmbeddingDataset(val_emb, val_labels, val_masks, val_conf)
        test_ds = EmbeddingDataset(test_emb, test_labels, test_masks, test_conf)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        if pos_weight is not None:
            pw = pos_weight.to(device)
        else:
            pw = None

        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                emb, conf, labels, masks = [b.to(device) for b in batch]
                optimizer.zero_grad()
                disease_logits, conf_pred = model(emb)
                disease_loss = masked_bce_loss(disease_logits, labels, masks, pw)
                adv_loss = torch.nn.functional.mse_loss(conf_pred, conf)
                loss = disease_loss + adv_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    emb, conf, labels, masks = [b.to(device) for b in batch]
                    disease_logits, conf_pred = model(emb)
                    vl = masked_bce_loss(disease_logits, labels, masks, pw)
                    val_loss += vl.item()
                    n_val += 1
            val_loss /= max(n_val, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate
        model.eval()
        all_logits, all_labels, all_masks = [], [], []
        all_conf_pred, all_conf_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                emb, conf, labels, masks = [b.to(device) for b in batch]
                disease_logits, conf_pred = model(emb)
                all_logits.append(disease_logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_masks.append(masks.cpu().numpy())
                all_conf_pred.append(conf_pred.cpu().numpy())
                all_conf_true.append(conf.cpu().numpy())

        logits = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels)
        masks_np = np.concatenate(all_masks)
        conf_pred = np.concatenate(all_conf_pred)
        conf_true = np.concatenate(all_conf_true)

        # Check adversary performance (gender AUROC)
        gender_idx = CONFOUNDERS.index("gender_binary")
        gender_pred = conf_pred[:, gender_idx]
        gender_true = conf_true[:, gender_idx]
        adv_gender_auroc = roc_auc_score(gender_true, gender_pred)

        disease_aurocs = compute_per_disease_auroc(logits, labels_np, masks_np, disease_names)
        mean_auroc = np.nanmean(list(disease_aurocs.values()))

        print(f"    alpha={alpha}: mean_AUROC={mean_auroc:.4f}, adv_gender={adv_gender_auroc:.3f}")

        all_alpha_results[alpha] = {
            "disease_aurocs": disease_aurocs,
            "mean_auroc": mean_auroc,
            "adv_gender_auroc": adv_gender_auroc,
        }

        # Select best alpha: adversary below target AND maximum disease AUROC
        if adv_gender_auroc < ADV_TARGET_AUROC:
            if best_result is None or mean_auroc > best_result["mean_auroc"]:
                best_result = all_alpha_results[alpha]
                best_alpha = alpha

    # If no alpha achieved target, use the one with lowest adversary AUROC
    # while still maintaining reasonable disease performance
    if best_result is None:
        # Find alpha with best trade-off
        candidates = sorted(all_alpha_results.items(),
                          key=lambda x: x[1]["adv_gender_auroc"])
        best_result = candidates[0][1]
        best_alpha = candidates[0][0]
        print(f"    (No alpha achieved target {ADV_TARGET_AUROC}; "
              f"using alpha={best_alpha} with adv_gender={best_result['adv_gender_auroc']:.3f})")

    return best_result["disease_aurocs"], best_alpha, all_alpha_results


def method_c_confounder_only(train_labels, train_masks, test_labels, test_masks,
                              train_conf, test_conf, disease_names, seed=SEED):
    """Method C: How well can confounders alone predict each disease?

    Returns:
        dict per disease: {disease: conf_auroc}
    """
    results = {}
    for j, disease in enumerate(disease_names):
        train_valid = train_masks[:, j].astype(bool)
        test_valid = test_masks[:, j].astype(bool)

        if test_valid.sum() < 30 or len(np.unique(test_labels[test_valid, j])) < 2:
            results[disease] = np.nan
            continue

        Z_train = train_conf[train_valid].copy()
        Z_test = test_conf[test_valid].copy()
        y_train = train_labels[train_valid, j]
        y_test = test_labels[test_valid, j]

        if len(np.unique(y_train)) < 2:
            results[disease] = np.nan
            continue

        # Impute NaN
        for col in range(Z_train.shape[1]):
            med = np.nanmedian(Z_train[:, col])
            Z_train[np.isnan(Z_train[:, col]), col] = med
            Z_test[np.isnan(Z_test[:, col]), col] = med

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
        clf.fit(Z_train, y_train)
        conf_pred = clf.predict_proba(Z_test)[:, 1]
        conf_auroc = roc_auc_score(y_test, conf_pred)
        results[disease] = conf_auroc

    return results


def classify_confounding(drop_a, cxr_added_value):
    """Classify confounding level based on Method A (primary).

    Uses config thresholds. Adversarial is supplementary only.
    """
    if np.isnan(drop_a) or np.isnan(cxr_added_value):
        return "unknown"

    # Genuine: small residualization drop AND meaningful CXR signal
    if drop_a < CONF_GENUINE_MAX_DROP and cxr_added_value > CONF_GENUINE_MIN_ADDED:
        return "genuine"

    # Heavily confounded: large drop OR CXR adds nothing beyond demographics
    if drop_a >= CONF_HEAVY_MIN_DROP or cxr_added_value < CONF_HEAVY_MAX_ADDED:
        return "heavily_confounded"

    # Partially confounded: everything in between
    return "partially_confounded"


def run_phase2(models=None):
    """Run full Phase 2 confounding analysis."""
    if models is None:
        models = MAIN_MODELS

    os.makedirs(RESULT_DIR, exist_ok=True)
    canonical_ids = load_canonical_ids()
    split = load_split()

    # Load metadata
    metadata = load_metadata()
    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    meta_df = metadata[metadata[ID_COL].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df[ID_COL].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)

    # Merge CheXpert
    merged_df, labels, masks, diseases = merge_chexpert(meta_df)
    tier1, tier2 = get_eligible_diseases(labels, masks, split, diseases)
    print(f"  Tier 1 diseases: {tier1}")
    print(f"  Tier 2 diseases: {tier2}")

    # Confounder values
    conf_cols = CONFOUNDERS
    conf_values = meta_df[conf_cols].values.astype(float)
    for j in range(conf_values.shape[1]):
        med = np.nanmedian(conf_values[split["train_idx"], j])
        conf_values[np.isnan(conf_values[:, j]), j] = med

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    all_results = []
    all_alpha_details = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Phase 2: {model_name}")
        print(f"{'='*60}")

        cxr_dim = MODELS[model_name]["embed_dim"]
        embeddings = get_aligned_embeddings(model_name, canonical_ids)

        # Scale embeddings
        train_emb, val_emb, test_emb, _ = scale_embeddings(embeddings, split)

        # Pos weight
        pw = compute_pos_weight(labels[train_idx], masks[train_idx])

        # --- Baseline: CXR-Only ---
        print("\n  [Baseline] CXR-Only")
        set_all_seeds(SEED)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cxr_model = CXROnlyModel(cxr_dim, len(diseases)).to(device)

        train_loader, val_loader, test_loader = make_loaders(
            train_emb, val_emb, test_emb,
            labels[train_idx], labels[val_idx], labels[test_idx],
            masks[train_idx], masks[val_idx], masks[test_idx],
        )
        cxr_model = train_disease_model(cxr_model, train_loader, val_loader, device, pw)
        logits, test_lab, test_msk = predict_disease(cxr_model, test_loader, device)
        original_aurocs = compute_per_disease_auroc(logits, test_lab, test_msk, diseases)
        orig_mean = np.nanmean(list(original_aurocs.values()))
        print(f"    Mean AUROC: {orig_mean:.4f}")

        # --- Method A: Residualization ---
        print("\n  [Method A] Linear Residualization")
        residual_emb, r2_explained = method_a_residualization(embeddings, conf_values, split)
        res_train, res_val, res_test, _ = scale_embeddings(residual_emb, split)

        set_all_seeds(SEED)
        res_model = CXROnlyModel(cxr_dim, len(diseases)).to(device)
        res_train_loader, res_val_loader, res_test_loader = make_loaders(
            res_train, res_val, res_test,
            labels[train_idx], labels[val_idx], labels[test_idx],
            masks[train_idx], masks[val_idx], masks[test_idx],
        )
        res_model = train_disease_model(res_model, res_train_loader, res_val_loader, device, pw)
        res_logits, _, _ = predict_disease(res_model, res_test_loader, device)
        residualized_aurocs = compute_per_disease_auroc(res_logits, test_lab, test_msk, diseases)
        res_mean = np.nanmean(list(residualized_aurocs.values()))
        print(f"    Mean AUROC: {res_mean:.4f} (drop: {orig_mean - res_mean:+.4f})")

        # --- Method B: Adversarial ---
        print("\n  [Method B] Adversarial Debiasing")
        conf_train = conf_values[train_idx].astype(np.float32)
        conf_val = conf_values[val_idx].astype(np.float32)
        conf_test = conf_values[test_idx].astype(np.float32)

        adv_aurocs, best_alpha, alpha_details = method_b_adversarial(
            train_emb, val_emb, test_emb,
            labels[train_idx], labels[val_idx], labels[test_idx],
            masks[train_idx], masks[val_idx], masks[test_idx],
            conf_train, conf_val, conf_test,
            cxr_dim, len(diseases), diseases,
            pos_weight=pw,
        )
        adv_mean = np.nanmean(list(adv_aurocs.values()))
        print(f"    Best alpha: {best_alpha}, Mean AUROC: {adv_mean:.4f}")

        # Save alpha sweep details
        for alpha, detail in alpha_details.items():
            all_alpha_details.append({
                "model": model_name,
                "alpha": alpha,
                "mean_disease_auroc": detail["mean_auroc"],
                "adv_gender_auroc": detail["adv_gender_auroc"],
            })

        # --- Method C: Confounder-Only baseline ---
        print("\n  [Method C] Confounder-Only Baseline")
        conf_only = method_c_confounder_only(
            labels[train_idx], masks[train_idx],
            labels[test_idx], masks[test_idx],
            conf_train, conf_test,
            diseases,
        )

        # Compile results per disease
        for disease in diseases:
            orig = original_aurocs.get(disease, np.nan)
            resid = residualized_aurocs.get(disease, np.nan)
            adv = adv_aurocs.get(disease, np.nan)
            conf_auc = conf_only.get(disease, np.nan)

            drop_a = orig - resid if not (np.isnan(orig) or np.isnan(resid)) else np.nan
            drop_b = orig - adv if not (np.isnan(orig) or np.isnan(adv)) else np.nan
            added = orig - conf_auc if not (np.isnan(orig) or np.isnan(conf_auc)) else np.nan

            # Classification: Method A primary
            classification = classify_confounding(drop_a, added)

            row = {
                "model": model_name,
                "disease": disease,
                "original_auroc": orig,
                "residualized_auroc": resid,
                "adversarial_auroc": adv,
                "confounder_only_auroc": conf_auc,
                "drop_residualization": drop_a,
                "drop_adversarial": drop_b,
                "cxr_added_value": added,
                "classification": classification,
                "adv_best_alpha": best_alpha,
                "embedding_r2_confounders": r2_explained,
            }
            all_results.append(row)

            tier = "T1" if disease in tier1 else "T2" if disease in tier2 else "--"
            if not np.isnan(orig):
                print(f"    [{tier}] {disease:30s}: orig={orig:.3f} resid={resid:.3f} "
                      f"adv={adv:.3f} conf={conf_auc:.3f} -> {classification}")
            else:
                print(f"    [{tier}] {disease:30s}: N/A")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "phase2_confounding.csv"), index=False)
    print(f"\n  Saved: {os.path.join(RESULT_DIR, 'phase2_confounding.csv')}")

    # Save alpha sweep details
    alpha_df = pd.DataFrame(all_alpha_details)
    alpha_df.to_csv(os.path.join(RESULT_DIR, "phase2_alpha_sweep.csv"), index=False)
    print(f"  Saved: phase2_alpha_sweep.csv")

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  Phase 2 Summary (Classification by Method A)")
    print(f"  {'='*60}")
    for cls in ["genuine", "partially_confounded", "heavily_confounded", "unknown"]:
        subset = results_df[results_df["classification"] == cls]
        if len(subset) > 0:
            print(f"  {cls:25s}: {len(subset)} disease-model pairs")
            for _, r in subset.iterrows():
                print(f"    {r['model']:30s} {r['disease']:30s} "
                      f"drop_A={r['drop_residualization']:+.3f} added={r['cxr_added_value']:+.3f}")

    # --- Threshold sensitivity analysis ---
    threshold_sensitivity(results_df)

    return results_df


def threshold_sensitivity(results_df):
    """Assess how confounding classifications change under different thresholds.

    Tests a grid of threshold values to demonstrate robustness (or lack thereof)
    of the genuine/partial/heavy classification.
    """
    print(f"\n  {'='*60}")
    print(f"  Threshold Sensitivity Analysis")
    print(f"  {'='*60}")

    drop_thresholds = [0.02, 0.03, 0.04, 0.05]
    added_thresholds = [0.03, 0.05, 0.07]
    heavy_drop_thresholds = [0.06, 0.08, 0.10]

    sensitivity_rows = []

    for gen_drop in drop_thresholds:
        for gen_added in added_thresholds:
            for heavy_drop in heavy_drop_thresholds:
                counts = {"genuine": 0, "partially_confounded": 0,
                          "heavily_confounded": 0, "unknown": 0}
                for _, row in results_df.iterrows():
                    drop_a = row["drop_residualization"]
                    added = row["cxr_added_value"]
                    if np.isnan(drop_a) or np.isnan(added):
                        counts["unknown"] += 1
                    elif drop_a < gen_drop and added > gen_added:
                        counts["genuine"] += 1
                    elif drop_a >= heavy_drop or added < 0.03:
                        counts["heavily_confounded"] += 1
                    else:
                        counts["partially_confounded"] += 1

                sensitivity_rows.append({
                    "genuine_max_drop": gen_drop,
                    "genuine_min_added": gen_added,
                    "heavy_min_drop": heavy_drop,
                    **counts,
                })

    sens_df = pd.DataFrame(sensitivity_rows)
    out_path = os.path.join(RESULT_DIR, "phase2_threshold_sensitivity.csv")
    sens_df.to_csv(out_path, index=False)
    print(f"  Tested {len(sens_df)} threshold combinations")
    print(f"  Genuine range: {sens_df['genuine'].min()}-{sens_df['genuine'].max()}")
    print(f"  Heavy range:   {sens_df['heavily_confounded'].min()}-{sens_df['heavily_confounded'].max()}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    run_phase2()
