"""
Full-scale embedding extraction for MIMIC-CXR (230K images).

Features:
- Incremental save: checkpoint every SAVE_EVERY images
- Resume: skip already-extracted images
- Configurable IMAGE_DIR (D: for full, C: for pilot)
- Progress logging with ETA
- Error handling: skip failed images, log them

Usage:
    python extract_full_mimic.py --model DINOv2-base
    python extract_full_mimic.py --model all
    python extract_full_mimic.py --model all --batch-size 32
"""
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.config import MODELS, EMBEDDING_DIR, MODEL_DIR, SEED

# ============================================================
# Configuration
# ============================================================
FULL_METADATA_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "full_mimic_cxr_metadata.csv"
)
# Full image directory: configurable via MIMIC_CXR_DIR env var.
# Default falls back to a local subfolder; set the env var to your image root
# (e.g., /path/to/mimic-cxr-jpg) before running.
FULL_IMAGE_DIR = os.environ.get(
    "MIMIC_CXR_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mimic-cxr-jpg"),
)

# Incremental save interval
SAVE_EVERY = 10000

# Output directory for full-scale embeddings (separate from pilot)
FULL_EMBEDDING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "embeddings_full"
)


def get_output_paths(model_name):
    """Get paths for embeddings, dicom_ids, and failed log."""
    os.makedirs(FULL_EMBEDDING_DIR, exist_ok=True)
    emb_path = os.path.join(FULL_EMBEDDING_DIR, f"{model_name}_embeddings.npy")
    ids_path = os.path.join(FULL_EMBEDDING_DIR, f"{model_name}_dicom_ids.npy")
    fail_path = os.path.join(FULL_EMBEDDING_DIR, f"{model_name}_failed.txt")
    return emb_path, ids_path, fail_path


def check_resume(model_name):
    """Check how many images have already been extracted."""
    emb_path, ids_path, _ = get_output_paths(model_name)
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        emb = np.load(emb_path)
        return emb.shape[0]
    return 0


def resolve_image_path(img_path, image_dir=None):
    """Resolve img_path to absolute path on HDD."""
    base = image_dir if image_dir is not None else FULL_IMAGE_DIR
    full = os.path.join(base, str(img_path))
    if os.path.exists(full):
        return full
    return None


# ============================================================
# Model loaders (returns model + transform/processor)
# ============================================================

def load_resnet50():
    import torchvision.models as models
    import torchvision.transforms as T
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().cuda()
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    def extract_batch(images):
        imgs = torch.stack([transform(img.convert("RGB")) for img in images]).cuda()
        with torch.no_grad():
            out = model(imgs).squeeze(-1).squeeze(-1)
        return out.cpu().numpy()
    return extract_batch, 2048


def load_dinov2():
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval().cuda()
    def extract_batch(images):
        images_rgb = [img.convert("RGB") for img in images]
        inputs = processor(images=images_rgb, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0].cpu().numpy()
    return extract_batch, 768


def load_xrv():
    import torchxrayvision as xrv
    import torchvision.transforms as T
    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model.eval().cuda()
    transform = T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.5], [0.25]),
    ])
    def extract_batch(images):
        imgs = []
        for img in images:
            gray = img.convert("L")
            rgb = Image.merge("RGB", [gray, gray, gray])
            t = transform(rgb).mean(dim=0, keepdim=True)
            imgs.append(t)
        batch = torch.stack(imgs).cuda()
        with torch.no_grad():
            features = model.features(batch)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            pooled = pooled.view(pooled.size(0), -1)
        return pooled.cpu().numpy()
    return extract_batch, 1024


def load_rad_dino():
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    model.eval().cuda()
    def extract_batch(images):
        images_rgb = [img.convert("RGB") for img in images]
        inputs = processor(images=images_rgb, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output.cpu().numpy()
    return extract_batch, 768


def load_biomedclip():
    from open_clip import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model.eval().cuda()
    def extract_batch(images):
        imgs = torch.stack([
            preprocess(img.convert("RGB")) for img in images
        ]).cuda()
        with torch.no_grad():
            features = model.encode_image(imgs)
        return features.cpu().numpy()
    return extract_batch, 512


def load_chexzero():
    import clip
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)
    ckpt_path = os.path.join(MODEL_DIR, "chexzero", "best_64_5e-05_original_22000_0.864.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        print(f"  Loaded CheXzero weights from {ckpt_path}")
    else:
        print(f"  WARNING: CheXzero weights not found, using base CLIP")
    model.eval()
    def extract_batch(images):
        imgs = torch.stack([
            preprocess(img.convert("RGB")) for img in images
        ]).to(device)
        with torch.no_grad():
            features = model.encode_image(imgs)
        return features.float().cpu().numpy()
    return extract_batch, 512


def load_clip_vit():
    """OpenAI CLIP ViT-B/16 — general-purpose VLM (no medical data in pretraining)."""
    from open_clip import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained(
        "ViT-B-16", pretrained="openai"
    )
    model.eval().cuda()
    def extract_batch(images):
        imgs = torch.stack([
            preprocess(img.convert("RGB")) for img in images
        ]).cuda()
        with torch.no_grad():
            features = model.encode_image(imgs)
        return features.float().cpu().numpy()
    return extract_batch, 512


def load_convnextv2():
    """ConvNeXtV2-Base (ImageNet-22k) — modern CNN with FCMAE pretraining."""
    from transformers import ConvNextV2Model, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-base-22k-224")
    model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")
    model.eval().cuda()
    def extract_batch(images):
        images_rgb = [img.convert("RGB") for img in images]
        inputs = processor(images=images_rgb, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output.squeeze(-1).squeeze(-1).cpu().numpy()
    return extract_batch, 1024


def load_gloria():
    """GLoRIA: CheXpert-pretrained CXR VLP model (ResNet-50 image encoder).

    Install: pip install git+https://github.com/marshuang80/gloria.git
    Weights: Download from https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh
             Place chexpert_resnet50.ckpt in ./pretrained/
    Pretraining: CheXpert images + Stanford reports (no MIMIC-CXR overlap)
    """
    try:
        import gloria
    except ImportError:
        raise ImportError(
            "GLoRIA not installed. Run:\n"
            "  pip install git+https://github.com/marshuang80/gloria.git\n"
            "Then download weights from:\n"
            "  https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh\n"
            "Place chexpert_resnet50.ckpt in ./pretrained/"
        )
    gloria_model = gloria.load_gloria(name="gloria_resnet50", device="cuda")
    gloria_model.eval()
    import torchvision.transforms as T
    # GLoRIA uses 'half' normalization: mean=std=0.5
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    # Detect actual embedding dim with dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).cuda()
        _, test_emb = gloria_model.image_encoder_forward(dummy)
        if test_emb.dim() > 2:
            test_emb = test_emb.squeeze(-1).squeeze(-1)
        actual_dim = test_emb.shape[1]
    print(f"  GLoRIA global embedding dim: {actual_dim}")
    def extract_batch(images):
        imgs = torch.stack([transform(img.convert("RGB")) for img in images]).cuda()
        with torch.no_grad():
            _, img_emb_g = gloria_model.image_encoder_forward(imgs)
        if img_emb_g.dim() > 2:
            img_emb_g = img_emb_g.squeeze(-1).squeeze(-1)
        return img_emb_g.cpu().numpy()
    return extract_batch, actual_dim


def load_chess():
    import torchvision.models as models
    import torchvision.transforms as T
    # Load ResNet50 backbone with 1-channel conv1 (grayscale CXR)
    model = models.resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Load MoCo v2 checkpoint and strip encoder_q prefix
    ckpt_path = os.path.join(MODEL_DIR, "chess", "chess_moco_resnet50.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    renamed = {}
    for k, v in sd.items():
        if k.startswith("module.encoder_q."):
            new_key = k[len("module.encoder_q."):]
            if not new_key.startswith("fc."):
                renamed[new_key] = v
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    fc_missing = [k for k in missing if k.startswith("fc.")]
    non_fc_missing = [k for k in missing if not k.startswith("fc.")]
    print(f"  CheSS loaded: {len(renamed)} keys, non-FC missing: {len(non_fc_missing)}")
    # Remove FC, keep backbone for 2048-d embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().cuda()
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.5], [0.25]),
    ])
    def extract_batch(images):
        imgs = []
        for img in images:
            gray = img.convert("L")
            t = transform(gray)  # (1, 224, 224)
            imgs.append(t)
        batch = torch.stack(imgs).cuda()
        with torch.no_grad():
            out = model(batch).squeeze(-1).squeeze(-1)
        return out.cpu().numpy()
    return extract_batch, 2048


MODEL_LOADERS = {
    "ResNet50-ImageNet": load_resnet50,
    "DINOv2-base": load_dinov2,
    "XRV-DenseNet-nih": load_xrv,
    "RAD-DINO": load_rad_dino,
    "BiomedCLIP": load_biomedclip,
    "CheXzero": load_chexzero,
    "CheSS-ResNet50": load_chess,
    "CLIP-ViT-B16": load_clip_vit,
    "ConvNeXtV2-Base": load_convnextv2,
    "GLoRIA": load_gloria,
}


# ============================================================
# Main extraction loop
# ============================================================

def extract_model(model_name, df, batch_size=32, image_dir=None):
    """Extract embeddings for one model with incremental save + resume."""
    emb_path, ids_path, fail_path = get_output_paths(model_name)
    embed_dim = MODELS[model_name]["embed_dim"]
    total = len(df)

    # Resume check
    already_done = check_resume(model_name)
    if already_done >= total:
        print(f"  {model_name}: already complete ({already_done}/{total})")
        return
    if already_done > 0:
        print(f"  Resuming {model_name}: {already_done}/{total} done")

    # Load model
    print(f"\n{'='*60}")
    print(f"  Loading model: {model_name}")
    print(f"{'='*60}")
    extract_batch_fn, expected_dim = MODEL_LOADERS[model_name]()
    assert expected_dim == embed_dim, f"Dim mismatch: {expected_dim} vs {embed_dim}"

    # Prepare image paths (skip already done)
    df_remaining = df.iloc[already_done:].reset_index(drop=True)
    img_paths = df_remaining["img_path"].tolist()
    dicom_ids = df_remaining["dicom_id"].tolist()

    # Load existing partial results
    if already_done > 0:
        all_emb = list(np.load(emb_path))
        all_ids = list(np.load(ids_path, allow_pickle=True))
    else:
        all_emb = []
        all_ids = []

    failed = []
    start_time = time.time()

    pbar = tqdm(range(0, len(img_paths), batch_size),
                desc=f"  {model_name}", total=(len(img_paths) + batch_size - 1) // batch_size)

    for batch_start in pbar:
        batch_paths = img_paths[batch_start:batch_start + batch_size]
        batch_dicoms = dicom_ids[batch_start:batch_start + batch_size]

        # Load images, track failures
        images = []
        valid_dicoms = []
        for p, d in zip(batch_paths, batch_dicoms):
            abs_path = resolve_image_path(p, image_dir=image_dir)
            if abs_path is None:
                failed.append(f"{d}\tfile_not_found\t{p}")
                continue
            try:
                img = Image.open(abs_path)
                img.load()  # force read
                images.append(img)
                valid_dicoms.append(d)
            except Exception as e:
                failed.append(f"{d}\tread_error\t{str(e)[:100]}")

        if not images:
            continue

        try:
            emb = extract_batch_fn(images)
            for i, d in enumerate(valid_dicoms):
                all_emb.append(emb[i])
                all_ids.append(d)
        except Exception as e:
            # If batch fails, try one by one
            for img, d in zip(images, valid_dicoms):
                try:
                    emb = extract_batch_fn([img])
                    all_emb.append(emb[0])
                    all_ids.append(d)
                except Exception as e2:
                    failed.append(f"{d}\textract_error\t{str(e2)[:100]}")

        # Close images to free memory
        for img in images:
            img.close()

        # Incremental save
        n_done = len(all_emb)
        if n_done > 0 and n_done % SAVE_EVERY < batch_size:
            _save_checkpoint(emb_path, ids_path, all_emb, all_ids, embed_dim)
            elapsed = time.time() - start_time
            rate = n_done / elapsed
            remaining = (total - already_done - n_done) / rate if rate > 0 else 0
            pbar.set_postfix({
                "saved": n_done,
                "failed": len(failed),
                "ETA": f"{remaining/60:.0f}min"
            })

    # Final save
    _save_checkpoint(emb_path, ids_path, all_emb, all_ids, embed_dim)

    # Save failed log
    if failed:
        with open(fail_path, "w") as f:
            f.write("\n".join(failed))
        print(f"  {len(failed)} images failed -> {fail_path}")

    elapsed = time.time() - start_time
    print(f"\n  {model_name} complete!")
    print(f"  Total extracted: {len(all_emb)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Time: {elapsed/60:.1f} min")


def _save_checkpoint(emb_path, ids_path, all_emb, all_ids, embed_dim):
    """Save current state to disk."""
    emb_arr = np.array(all_emb, dtype=np.float32)
    ids_arr = np.array(all_ids)
    np.save(emb_path, emb_arr)
    np.save(ids_path, ids_arr)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full-scale MIMIC-CXR embedding extraction")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all' (default: all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--image-dir", type=str, default=FULL_IMAGE_DIR,
                        help=f"Image directory (default: {FULL_IMAGE_DIR})")
    parser.add_argument("--csv", type=str, default=FULL_METADATA_CSV,
                        help="Metadata CSV path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check file availability without extracting")
    args = parser.parse_args()

    # Update image dir if specified via CLI
    _image_dir = args.image_dir

    print(f"Image dir: {_image_dir}")
    print(f"Metadata: {args.csv}")
    print(f"Output: {FULL_EMBEDDING_DIR}")
    print()

    # Load metadata
    df = pd.read_csv(args.csv)
    print(f"Total images: {len(df)}")

    # Check image availability
    n_found = 0
    for p in tqdm(df["img_path"].head(1000), desc="  Checking images (sample)"):
        if resolve_image_path(p, image_dir=_image_dir) is not None:
            n_found += 1
    avail_pct = n_found / min(1000, len(df)) * 100
    print(f"  Image availability (sample): {avail_pct:.1f}%")

    if avail_pct < 50:
        print(f"\n  WARNING: Only {avail_pct:.1f}% of images found!")
        print(f"  Make sure images are downloaded to {_image_dir}")
        if not args.dry_run:
            resp = input("  Continue anyway? (y/N): ")
            if resp.lower() != "y":
                return

    if args.dry_run:
        print("\n  Dry run complete.")
        return

    # Select models
    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]
        if args.model not in MODELS:
            print(f"  ERROR: Unknown model '{args.model}'")
            print(f"  Available: {list(MODELS.keys())}")
            return

    # Extract
    for model_name in models:
        extract_model(model_name, df, batch_size=args.batch_size, image_dir=_image_dir)

    print("\n" + "="*60)
    print("  ALL DONE!")
    print("="*60)
    for model_name in models:
        emb_path, _, _ = get_output_paths(model_name)
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            print(f"  {model_name}: {emb.shape}")


if __name__ == "__main__":
    main()
