"""
Unified embedding extraction for all 6 foundation models.
Each extractor returns (embeddings, dicom_ids) as numpy arrays.
"""
import os
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

from experiments.config import (
    IMAGE_DIR, EMBEDDING_DIR, MODEL_DIR, MODELS, SEED,
)


def _load_image_paths(df):
    """Get absolute image paths from dataframe."""
    paths = []
    for p in df["img_path"]:
        full = os.path.join(IMAGE_DIR, str(p))
        if os.path.exists(full):
            paths.append(full)
        else:
            paths.append(None)
    return paths


# ============================================================
# 1. ResNet50-ImageNet
# ============================================================
def extract_resnet50(df, batch_size=64):
    """Extract ResNet50-ImageNet embeddings (2048-d)."""
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader

    print("  Loading ResNet50-ImageNet...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove FC
    model.eval().cuda()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class ImgDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), idx

    valid_mask = [p is not None for p in _load_image_paths(df)]
    valid_paths = [p for p in _load_image_paths(df) if p is not None]
    loader = DataLoader(
        ImgDataset(valid_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    all_emb = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="  ResNet50-ImageNet"):
            out = model(imgs.cuda()).squeeze(-1).squeeze(-1)
            all_emb.append(out.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    dicom_ids = df.loc[valid_mask, "dicom_id"].values
    return embeddings, dicom_ids


# ============================================================
# 2. DINOv2-base (facebook/dinov2-base)
# ============================================================
def extract_dinov2(df, batch_size=32):
    """Extract DINOv2-base embeddings (768-d CLS token)."""
    from transformers import AutoModel, AutoImageProcessor

    print("  Loading DINOv2-base...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval().cuda()

    valid_paths = [p for p in _load_image_paths(df) if p is not None]
    valid_mask = [p is not None for p in _load_image_paths(df)]

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_paths), batch_size), desc="  DINOv2-base"):
            batch_paths = valid_paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0]  # CLS token (batch, 768)
            all_emb.append(cls_emb.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    dicom_ids = df.loc[valid_mask, "dicom_id"].values
    return embeddings, dicom_ids


# ============================================================
# 3. TorchXRayVision DenseNet
# ============================================================
def extract_xrv(df, weights="nih", batch_size=64):
    """Extract TorchXRayVision DenseNet embeddings (1024-d)."""
    import torchxrayvision as xrv
    import torchvision.transforms as T

    print(f"  Loading XRV DenseNet ({weights})...")
    model = xrv.models.DenseNet(weights=f"densenet121-res224-{weights}")
    model.eval().cuda()

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5], [0.25]),
    ])

    valid_paths = [p for p in _load_image_paths(df) if p is not None]
    valid_mask = [p is not None for p in _load_image_paths(df)]

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_paths), batch_size), desc=f"  XRV-{weights}"):
            batch_paths = valid_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("L")
                img_rgb = Image.merge("RGB", [img, img, img])
                t = transform(img_rgb)
                # XRV expects single channel: mean across RGB
                t = t.mean(dim=0, keepdim=True)
                imgs.append(t)
            batch = torch.stack(imgs).cuda()
            features = model.features(batch)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            pooled = pooled.view(pooled.size(0), -1)
            all_emb.append(pooled.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    dicom_ids = df.loc[valid_mask, "dicom_id"].values
    return embeddings, dicom_ids


# ============================================================
# 4. RAD-DINO (microsoft/rad-dino)
# ============================================================
def extract_rad_dino(df, batch_size=32):
    """Extract RAD-DINO embeddings (768-d CLS token)."""
    from transformers import AutoModel, AutoImageProcessor

    print("  Loading RAD-DINO...")
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    model.eval().cuda()

    valid_paths = [p for p in _load_image_paths(df) if p is not None]
    valid_mask = [p is not None for p in _load_image_paths(df)]

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_paths), batch_size), desc="  RAD-DINO"):
            batch_paths = valid_paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_emb = outputs.pooler_output  # (batch, 768)
            all_emb.append(cls_emb.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    dicom_ids = df.loc[valid_mask, "dicom_id"].values
    return embeddings, dicom_ids


# ============================================================
# 5. BiomedCLIP (microsoft/BiomedCLIP)
# ============================================================
def extract_biomedclip(df, batch_size=64):
    """Extract BiomedCLIP image embeddings (512-d)."""
    from open_clip import create_model_from_pretrained

    print("  Loading BiomedCLIP...")
    model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model.eval().cuda()

    valid_paths = [p for p in _load_image_paths(df) if p is not None]
    valid_mask = [p is not None for p in _load_image_paths(df)]

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_paths), batch_size), desc="  BiomedCLIP"):
            batch_paths = valid_paths[i:i+batch_size]
            imgs = torch.stack([
                preprocess(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).cuda()
            features = model.encode_image(imgs)
            all_emb.append(features.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    dicom_ids = df.loc[valid_mask, "dicom_id"].values
    return embeddings, dicom_ids


# ============================================================
# 6. CheXzero
# ============================================================
def extract_chexzero(df, batch_size=64):
    """Extract CheXzero image embeddings (512-d).

    Requires: pip install git+https://github.com/openai/CLIP.git
    Weights must be downloaded to cxr_foundation_models/chexzero/
    """
    import clip

    print("  Loading CheXzero...")
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load CheXzero fine-tuned weights
    ckpt_path = os.path.join(MODEL_DIR, "chexzero", "best_64_5e-05_original_22000_0.864.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f"  Loaded CheXzero weights from {ckpt_path}")
    else:
        print(f"  WARNING: CheXzero weights not found at {ckpt_path}")
        print(f"  Using base CLIP ViT-B/32 (not fine-tuned)")

    model.eval()

    valid_paths = [p for p in _load_image_paths(df) if p is not None]
    valid_mask = [p is not None for p in _load_image_paths(df)]

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_paths), batch_size), desc="  CheXzero"):
            batch_paths = valid_paths[i:i+batch_size]
            imgs = torch.stack([
                preprocess(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(device)
            features = model.encode_image(imgs)
            all_emb.append(features.float().cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    dicom_ids = df.loc[valid_mask, "dicom_id"].values
    return embeddings, dicom_ids



# ============================================================
# Unified extraction interface
# ============================================================
EXTRACTORS = {
    "ResNet50-ImageNet": extract_resnet50,
    "DINOv2-base": extract_dinov2,
    "XRV-DenseNet-nih": extract_xrv,
    "RAD-DINO": extract_rad_dino,
    "BiomedCLIP": extract_biomedclip,
    "CheXzero": extract_chexzero,
}


def extract_and_cache(model_name, df, force=False):
    """Extract embeddings for a model, caching to disk.

    Returns:
        embeddings: np.array (n, embed_dim)
        dicom_ids: np.array of dicom_id strings
    """
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    emb_path = os.path.join(EMBEDDING_DIR, f"{model_name}_embeddings.npy")
    ids_path = os.path.join(EMBEDDING_DIR, f"{model_name}_dicom_ids.npy")

    if os.path.exists(emb_path) and os.path.exists(ids_path) and not force:
        print(f"  Loading cached: {model_name}")
        embeddings = np.load(emb_path)
        dicom_ids = np.load(ids_path, allow_pickle=True)
        print(f"  Shape: {embeddings.shape}")
        return embeddings, dicom_ids

    print(f"\n{'='*60}")
    print(f"  Extracting: {model_name}")
    print(f"{'='*60}")

    extractor = EXTRACTORS[model_name]
    embeddings, dicom_ids = extractor(df)

    np.save(emb_path, embeddings)
    np.save(ids_path, dicom_ids)
    print(f"  Cached: {emb_path} ({embeddings.shape})")

    return embeddings, dicom_ids


def extract_all(df, models=None, force=False):
    """Extract embeddings for all (or specified) models.

    Returns:
        dict of {model_name: (embeddings, dicom_ids)}
    """
    if models is None:
        models = list(MODELS.keys())

    results = {}
    for model_name in models:
        emb, ids = extract_and_cache(model_name, df, force=force)
        results[model_name] = (emb, ids)

    return results


def migrate_legacy_embeddings(df):
    """Migrate embeddings from the old flat directory to embeddings/ dir.

    Checks for legacy files like xrv_embeddings.npy in PROJECT_DIR
    and copies them to EMBEDDING_DIR with canonical names.
    """
    from experiments.config import PROJECT_DIR

    legacy_map = {
        "phase2_xrv_embeddings.npy": ("XRV-DenseNet-nih", "phase2_xrv_dicom_ids.npy"),
    }

    for old_emb, (new_name, old_ids) in legacy_map.items():
        old_emb_path = os.path.join(PROJECT_DIR, old_emb)
        new_emb_path = os.path.join(EMBEDDING_DIR, f"{new_name}_embeddings.npy")
        new_ids_path = os.path.join(EMBEDDING_DIR, f"{new_name}_dicom_ids.npy")

        if os.path.exists(old_emb_path) and not os.path.exists(new_emb_path):
            import shutil
            os.makedirs(EMBEDDING_DIR, exist_ok=True)
            shutil.copy2(old_emb_path, new_emb_path)
            print(f"  Migrated: {old_emb} -> {new_name}_embeddings.npy")

            if old_ids:
                old_ids_path = os.path.join(PROJECT_DIR, old_ids)
                if os.path.exists(old_ids_path):
                    shutil.copy2(old_ids_path, new_ids_path)
