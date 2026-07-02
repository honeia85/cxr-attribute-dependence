"""
Neural network model definitions for probing, fusion, and adversarial experiments.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from sklearn.metrics import roc_auc_score

from experiments.config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    SCHEDULER_PATIENCE, SCHEDULER_FACTOR, MIN_LR, GRAD_CLIP,
)


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Datasets
# ============================================================
class EmbeddingDataset(Dataset):
    """Dataset for embedding + optional metadata + labels."""
    def __init__(self, embeddings, labels=None, masks=None, metadata=None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.masks = torch.BoolTensor(masks) if masks is not None else None
        self.metadata = torch.FloatTensor(metadata) if metadata is not None else None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        items = [self.embeddings[idx]]
        if self.metadata is not None:
            items.append(self.metadata[idx])
        else:
            items.append(torch.zeros(1))  # placeholder
        if self.labels is not None:
            items.append(self.labels[idx])
            items.append(self.masks[idx])
        return tuple(items)


# ============================================================
# Probing Models
# ============================================================
class MLPProbe(nn.Module):
    """MLP probe for single-target prediction.

    Hidden dimension scales proportionally to input_dim to ensure
    fair comparison across models with different embedding sizes.
    """
    def __init__(self, input_dim, task="binary"):
        super().__init__()
        self.task = task
        hidden_dim = max(128, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Disease Prediction Models (Phase 2/4)
# ============================================================
class CXROnlyModel(nn.Module):
    """CXR embeddings -> disease predictions.

    Hidden dimensions scale proportionally to input_dim to ensure
    fair comparison across models with different embedding sizes.
    """
    def __init__(self, cxr_dim, num_labels, **kw):
        super().__init__()
        h1 = max(256, cxr_dim // 2)
        h2 = max(128, h1 // 2)
        self.net = nn.Sequential(
            nn.Linear(cxr_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h2, num_labels),
        )

    def forward(self, cxr_emb, metadata=None):
        return self.net(cxr_emb)


class MetadataOnlyModel(nn.Module):
    """Metadata -> disease predictions.

    Hidden dimensions scale proportionally, clamped to reasonable range
    given the small input dimension (3-25 features).
    """
    def __init__(self, meta_dim, num_labels, **kw):
        super().__init__()
        h1 = max(64, min(128, meta_dim * 4))
        h2 = max(32, h1 // 2)
        self.net = nn.Sequential(
            nn.Linear(meta_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h2, num_labels),
        )

    def forward(self, cxr_emb=None, metadata=None):
        return self.net(metadata)


class LateFusionModel(nn.Module):
    """CXR + Metadata late fusion.

    CXR branch scales proportionally to cxr_dim (matching CXROnlyModel).
    Metadata branch scales proportionally to meta_dim.
    """
    def __init__(self, cxr_dim, meta_dim, num_labels, **kw):
        super().__init__()
        cxr_h = max(256, cxr_dim // 2)
        meta_h1 = max(64, min(128, meta_dim * 4))
        meta_h2 = max(32, meta_h1 // 2)
        self.cxr_branch = nn.Sequential(
            nn.Linear(cxr_dim, cxr_h),
            nn.BatchNorm1d(cxr_h),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, meta_h1),
            nn.BatchNorm1d(meta_h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_h1, meta_h2),
            nn.BatchNorm1d(meta_h2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        head_dim = cxr_h + meta_h2
        head_hidden = max(128, head_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(head_dim, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(head_hidden, num_labels),
        )

    def forward(self, cxr_emb, metadata):
        cxr_h = self.cxr_branch(cxr_emb)
        meta_h = self.meta_branch(metadata)
        return self.head(torch.cat([cxr_h, meta_h], dim=1))


# ============================================================
# Adversarial Debiasing Model (Phase 2 Method B)
# ============================================================
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class AdversarialModel(nn.Module):
    """Disease classifier with adversarial demographic predictor.

    Backbone dimensions scale proportionally to cxr_dim (matching CXROnlyModel).
    """
    def __init__(self, cxr_dim, num_labels, num_confounders=3):
        super().__init__()
        h1 = max(256, cxr_dim // 2)
        h2 = max(128, h1 // 2)
        self.backbone = nn.Sequential(
            nn.Linear(cxr_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # Disease head
        self.disease_head = nn.Linear(h2, num_labels)
        # Adversary head (predicts confounders)
        self.adversary = nn.Sequential(
            nn.Linear(h2, 64),
            nn.ReLU(),
            nn.Linear(64, num_confounders),
        )
        self.adv_alpha = 1.0

    def forward(self, cxr_emb, metadata=None):
        h = self.backbone(cxr_emb)
        disease_logits = self.disease_head(h)
        h_rev = GradientReversal.apply(h, self.adv_alpha)
        confounder_pred = self.adversary(h_rev)
        return disease_logits, confounder_pred


# ============================================================
# Training Utilities
# ============================================================
def compute_pos_weight(labels, masks):
    """Compute positive class weights for imbalanced diseases."""
    weights = []
    for j in range(labels.shape[1]):
        valid = masks[:, j]
        if valid.sum() == 0:
            weights.append(1.0)
            continue
        n_pos = labels[valid, j].sum()
        n_neg = valid.sum() - n_pos
        w = min(n_neg / n_pos, 10.0) if n_pos > 0 else 1.0
        weights.append(max(w, 0.1))
    return torch.FloatTensor(weights)


def masked_bce_loss(logits, targets, mask, pos_weight=None):
    """BCE loss with masking for uncertain labels."""
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    masked_loss = loss * mask.float()
    num_valid = mask.float().sum()
    if num_valid > 0:
        return masked_loss.sum() / num_valid
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def train_disease_model(model, train_loader, val_loader, device,
                        pos_weight=None, epochs=EPOCHS, lr=LR,
                        patience=PATIENCE):
    """Standard training loop for disease prediction models.

    Returns:
        model: trained model (best val loss checkpoint)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR, min_lr=MIN_LR
    )
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, n_batches = 0, 0
        for batch in train_loader:
            emb, meta, labels, masks = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(emb, meta)
            loss = masked_bce_loss(logits, labels, masks, pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        val_loss, n_val = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                emb, meta, labels, masks = [b.to(device) for b in batch]
                logits = model(emb, meta)
                val_loss += masked_bce_loss(logits, labels, masks, pos_weight).item()
                n_val += 1

        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_disease(model, loader, device):
    """Get predictions from a disease model.

    Returns:
        logits, labels, masks: all as numpy arrays
    """
    model.eval()
    all_logits, all_labels, all_masks = [], [], []
    for batch in loader:
        emb, meta, labels, masks = [b.to(device) for b in batch]
        logits = model(emb, meta)
        if isinstance(logits, tuple):
            logits = logits[0]  # adversarial model returns (disease, confounder)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    return (np.concatenate(all_logits),
            np.concatenate(all_labels),
            np.concatenate(all_masks))


def compute_per_disease_auroc(logits, labels, masks, disease_names):
    """Compute AUROC per disease.

    Returns:
        dict of {disease_name: auroc} (NaN if insufficient data)
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    results = {}
    for j, name in enumerate(disease_names):
        valid = masks[:, j].astype(bool)
        if valid.sum() < 20 or len(np.unique(labels[valid, j])) < 2:
            results[name] = np.nan
            continue
        results[name] = roc_auc_score(labels[valid, j], probs[valid, j])
    return results


def make_loaders(train_emb, val_emb, test_emb,
                 train_labels, val_labels, test_labels,
                 train_masks, val_masks, test_masks,
                 train_meta=None, val_meta=None, test_meta=None,
                 batch_size=BATCH_SIZE):
    """Create DataLoaders for train/val/test."""
    train_ds = EmbeddingDataset(train_emb, train_labels, train_masks, train_meta)
    val_ds = EmbeddingDataset(val_emb, val_labels, val_masks, val_meta)
    test_ds = EmbeddingDataset(test_emb, test_labels, test_masks, test_meta)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
