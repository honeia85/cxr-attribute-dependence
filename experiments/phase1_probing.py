"""
Phase 1: Linear Probing - What clinical information do CXR FMs encode?

For each model and each metadata target:
  - Linear probe (LogisticRegression / Ridge)
  - MLP probe (5 seeds)
  - Bootstrap CI on test set
  - DeLong pairwise comparison across models
  - Permutation null for borderline targets
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from experiments.config import (
    MODELS, MAIN_MODELS, SEED, MULTI_SEEDS, RESULT_DIR, N_BOOTSTRAP,
    EPOCHS, LR, PATIENCE, ID_COL,
)
from experiments.data import (
    load_metadata, load_canonical_ids, load_split,
    get_aligned_embeddings, get_probing_targets,
)
from experiments.stats import (
    bootstrap_ci, bootstrap_auroc, bootstrap_auprc,
    delong_test, benjamini_hochberg, permutation_auroc, format_ci,
)
from experiments.models import MLPProbe, set_all_seeds


def run_linear_probe(X_train, y_train, X_test, y_test, task="binary"):
    """Run a single linear probe.

    Returns:
        dict with metrics and predictions
    """
    if task == "binary":
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        auroc_pt, auroc_lo, auroc_hi = bootstrap_auroc(y_test, y_pred)
        auprc_pt, auprc_lo, auprc_hi = bootstrap_auprc(y_test, y_pred)
        return {
            "auroc": auroc_pt, "auroc_ci_lo": auroc_lo, "auroc_ci_hi": auroc_hi,
            "auprc": auprc_pt, "auprc_ci_lo": auprc_lo, "auprc_ci_hi": auprc_hi,
            "y_pred": y_pred,
        }
    else:
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Bootstrap CI for MAE
        mae_pt, mae_lo, mae_hi = bootstrap_ci(
            y_test, y_pred, mean_absolute_error
        )
        r2_pt, r2_lo, r2_hi = bootstrap_ci(
            y_test, y_pred, r2_score
        )
        return {
            "mae": mae_pt, "mae_ci_lo": mae_lo, "mae_ci_hi": mae_hi,
            "r2": r2_pt, "r2_ci_lo": r2_lo, "r2_ci_hi": r2_hi,
            "y_pred": y_pred,
        }


def run_mlp_probe(X_train, y_train, X_val, y_val, X_test, y_test,
                  task="binary", seeds=None):
    """Run MLP probe with multiple seeds.

    Returns:
        dict with mean/std metrics and ensemble predictions
    """
    if seeds is None:
        seeds = MULTI_SEEDS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_preds = []
    all_metrics = []

    # Normalize regression targets for MLP training
    y_mean, y_std = 0.0, 1.0
    if task == "regression":
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std < 1e-8:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std
    else:
        y_train_norm = y_train
        y_val_norm = y_val

    for seed in seeds:
        set_all_seeds(seed)
        model = MLPProbe(X_train.shape[1], task=task).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        X_tr_t = torch.FloatTensor(X_train).to(device)
        y_tr_t = torch.FloatTensor(y_train_norm).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val_norm).to(device)
        X_te_t = torch.FloatTensor(X_test).to(device)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            pred = model(X_tr_t)
            if task == "binary":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y_tr_t)
            else:
                loss = torch.nn.functional.mse_loss(pred, y_tr_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                if task == "binary":
                    val_loss = torch.nn.functional.binary_cross_entropy_with_logits(val_pred, y_val_t).item()
                else:
                    val_loss = torch.nn.functional.mse_loss(val_pred, y_val_t).item()

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

        model.eval()
        with torch.no_grad():
            test_pred = model(X_te_t).cpu().numpy()

        if task == "binary":
            test_prob = 1.0 / (1.0 + np.exp(-test_pred))
            auroc = roc_auc_score(y_test, test_prob)
            all_metrics.append(auroc)
            all_preds.append(test_prob)
        else:
            # Inverse transform to original scale
            test_pred_orig = test_pred * y_std + y_mean
            mae = mean_absolute_error(y_test, test_pred_orig)
            all_metrics.append(mae)
            all_preds.append(test_pred_orig)

    # Ensemble: mean of predictions
    ensemble_pred = np.mean(all_preds, axis=0)
    metrics_arr = np.array(all_metrics)

    if task == "binary":
        ens_auroc = roc_auc_score(y_test, ensemble_pred)
        return {
            "auroc_mean": metrics_arr.mean(),
            "auroc_std": metrics_arr.std(),
            "auroc_ensemble": ens_auroc,
            "y_pred": ensemble_pred,
        }
    else:
        ens_mae = mean_absolute_error(y_test, ensemble_pred)
        return {
            "mae_mean": metrics_arr.mean(),
            "mae_std": metrics_arr.std(),
            "mae_ensemble": ens_mae,
            "y_pred": ensemble_pred,
        }


def run_phase1(models=None):
    """Run full Phase 1 probing analysis.

    Returns:
        DataFrame with all results
    """
    if models is None:
        models = MAIN_MODELS

    os.makedirs(RESULT_DIR, exist_ok=True)

    canonical_ids = load_canonical_ids()
    split = load_split()
    metadata = load_metadata()

    # Align metadata to canonical order
    id_to_order = {did: i for i, did in enumerate(canonical_ids)}
    meta_df = metadata[metadata[ID_COL].isin(set(canonical_ids))].copy()
    meta_df["_order"] = meta_df[ID_COL].map(id_to_order)
    meta_df = meta_df.sort_values("_order").reset_index(drop=True)

    targets = get_probing_targets(meta_df)
    print(f"\n  Probing targets: {len(targets)} "
          f"({sum(1 for t in targets if t['task']=='binary')} binary, "
          f"{sum(1 for t in targets if t['task']=='regression')} regression)")

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    all_results = []
    all_predictions = {}  # for DeLong tests: {(model, target): y_pred}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Phase 1: {model_name}")
        print(f"{'='*60}")

        embeddings = get_aligned_embeddings(model_name, canonical_ids)

        # StandardScale embeddings
        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_idx])
        X_val = scaler.transform(embeddings[val_idx])
        X_test = scaler.transform(embeddings[test_idx])

        for target in targets:
            name = target["name"]
            task = target["task"]
            y = target["y"]
            mask = target["mask"]

            # Get masked data for each split
            train_mask = mask[train_idx]
            val_mask = mask[val_idx]
            test_mask = mask[test_idx]

            X_tr = X_train[train_mask]
            X_va = X_val[val_mask]
            X_te = X_test[test_mask]
            y_tr = y[train_idx][train_mask]
            y_va = y[val_idx][val_mask]
            y_te = y[test_idx][test_mask]

            if len(X_te) < 30:
                continue

            # Linear probe
            linear_res = run_linear_probe(X_tr, y_tr, X_te, y_te, task)

            # MLP probe
            mlp_res = run_mlp_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, task)

            # Store predictions for DeLong
            all_predictions[(model_name, name)] = linear_res["y_pred"]

            # Permutation test for borderline binary targets
            perm_p = np.nan
            if task == "binary" and 0.55 <= linear_res["auroc"] <= 0.65:
                _, perm_p, _, _ = permutation_auroc(y_te, linear_res["y_pred"])

            # Build result row
            row = {
                "model": model_name,
                "target": name,
                "task": task,
                "n_test": len(y_te),
            }
            if task == "binary":
                row.update({
                    "prevalence": target.get("prevalence", np.nan),
                    "linear_auroc": linear_res["auroc"],
                    "linear_auroc_ci_lo": linear_res["auroc_ci_lo"],
                    "linear_auroc_ci_hi": linear_res["auroc_ci_hi"],
                    "linear_auprc": linear_res["auprc"],
                    "mlp_auroc_mean": mlp_res["auroc_mean"],
                    "mlp_auroc_std": mlp_res["auroc_std"],
                    "mlp_auroc_ensemble": mlp_res["auroc_ensemble"],
                    "perm_p": perm_p,
                })
                print(f"  {name:30s}: AUROC={format_ci(linear_res['auroc'], linear_res['auroc_ci_lo'], linear_res['auroc_ci_hi'])}"
                      f"  MLP={mlp_res['auroc_mean']:.3f}+/-{mlp_res['auroc_std']:.3f}")
            else:
                row.update({
                    "linear_mae": linear_res["mae"],
                    "linear_mae_ci_lo": linear_res["mae_ci_lo"],
                    "linear_mae_ci_hi": linear_res["mae_ci_hi"],
                    "linear_r2": linear_res["r2"],
                    "mlp_mae_mean": mlp_res["mae_mean"],
                    "mlp_mae_std": mlp_res["mae_std"],
                })
                print(f"  {name:30s}: MAE={format_ci(linear_res['mae'], linear_res['mae_ci_lo'], linear_res['mae_ci_hi'])}"
                      f"  R2={linear_res['r2']:.3f}")

            all_results.append(row)

    # DeLong pairwise comparisons
    print(f"\n{'='*60}")
    print(f"  DeLong Pairwise Comparisons")
    print(f"{'='*60}")

    delong_results = []
    binary_targets = [t["name"] for t in targets if t["task"] == "binary"]

    for target_name in binary_targets:
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                key_a = (model_a, target_name)
                key_b = (model_b, target_name)
                if key_a not in all_predictions or key_b not in all_predictions:
                    continue

                y_te = targets[[t["name"] for t in targets].index(target_name)]["y"]
                test_mask = targets[[t["name"] for t in targets].index(target_name)]["mask"][test_idx]
                y_te = y_te[test_idx][test_mask]

                auroc_a, auroc_b, z, p = delong_test(
                    y_te, all_predictions[key_a], all_predictions[key_b]
                )
                delong_results.append({
                    "target": target_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "auroc_a": auroc_a,
                    "auroc_b": auroc_b,
                    "z_stat": z,
                    "p_value": p,
                })

    # FDR correction
    if delong_results:
        delong_df = pd.DataFrame(delong_results)
        p_vals = delong_df["p_value"].values
        rejected, adjusted_p = benjamini_hochberg(p_vals)
        delong_df["p_adjusted"] = adjusted_p
        delong_df["significant"] = rejected
        delong_df.to_csv(os.path.join(RESULT_DIR, "phase1_delong.csv"), index=False)
        print(f"  {rejected.sum()} / {len(rejected)} comparisons significant after FDR correction")

    # Save main results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULT_DIR, "phase1_probing.csv"), index=False)
    print(f"\n  Saved: {os.path.join(RESULT_DIR, 'phase1_probing.csv')}")

    return results_df


if __name__ == "__main__":
    run_phase1()
