"""
CheXpert Plus chunk-by-chunk embedding extraction pipeline.

Strategy:
  1. Download one PNG ZIP chunk from Azure via AzCopy
  2. Extract embeddings directly from ZIP (streaming, no full unzip)
  3. Delete ZIP after extraction
  4. Repeat for all 5 chunks
  5. Merge embeddings across chunks

Usage:
    # Process all chunks sequentially
    python extract_chexpert_plus.py --model all

    # Process single model, single chunk (for testing)
    python extract_chexpert_plus.py --model DINOv2-base --chunk 4

    # Skip download (if ZIP already on disk)
    python extract_chexpert_plus.py --model DINOv2-base --chunk 0 --skip-download

    # Dry run (check setup, don't download)
    python extract_chexpert_plus.py --dry-run
"""
import sys
import os
import time
import argparse
import subprocess
import zipfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.config import MODELS, MODEL_DIR

# ============================================================
# Configuration
# ============================================================
# Path to the azcopy binary. Configurable via AZCOPY_BIN env var
# (default assumes it is on PATH).
AZCOPY = os.environ.get("AZCOPY_BIN", "azcopy")

# SAS URL for CheXpert Plus container. Stanford AIMI requires you to obtain
# your own credentialed SAS URL from https://stanfordaimi.azurewebsites.net
# and set it via the CHEXPERT_PLUS_SAS_URL env var.
SAS_URL = os.environ.get(
    "CHEXPERT_PLUS_SAS_URL",
    "https://aimistanforddatasets01.blob.core.windows.net/chexpertplus?<YOUR_SAS_TOKEN_HERE>",
)

# PNG chunk info
CHUNKS = {
    0: "PNG/png_chexpert_plus_chunk_0.zip",
    1: "PNG/png_chexpert_plus_chunk_1.zip",
    2: "PNG/png_chexpert_plus_chunk_2.zip",
    3: "PNG/png_chexpert_plus_chunk_3.zip",
    4: "PNG/png_chexpert_plus_chunk_4.zip",
}

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_CSV = os.path.join(PROJECT_DIR, "chexpert_plus_metadata.csv")
TEMP_DIR = os.path.join(PROJECT_DIR, "chexpert_temp")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "embeddings_chexpert")

SAVE_EVERY = 5000  # checkpoint interval


def get_chunk_url(chunk_id):
    """Build full SAS URL for a specific chunk."""
    base = SAS_URL.split("?")[0]
    sas = SAS_URL.split("?")[1]
    blob_path = CHUNKS[chunk_id]
    return f"{base}/{blob_path}?{sas}"


def download_chunk(chunk_id):
    """Download a chunk ZIP via AzCopy."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    url = get_chunk_url(chunk_id)
    local_path = os.path.join(TEMP_DIR, f"png_chexpert_plus_chunk_{chunk_id}.zip")

    if os.path.exists(local_path):
        size_gb = os.path.getsize(local_path) / (1024**3)
        print(f"  Chunk {chunk_id} already on disk: {size_gb:.1f} GB")
        return local_path

    print(f"  Downloading chunk {chunk_id}...")
    print(f"  URL: {CHUNKS[chunk_id]}")
    print(f"  Destination: {local_path}")

    cmd = [AZCOPY, "copy", url, local_path]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"AzCopy failed with code {result.returncode}")

    size_gb = os.path.getsize(local_path) / (1024**3)
    print(f"  Downloaded: {size_gb:.1f} GB")
    return local_path


def delete_chunk(chunk_id):
    """Delete a downloaded chunk ZIP."""
    local_path = os.path.join(TEMP_DIR, f"png_chexpert_plus_chunk_{chunk_id}.zip")
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"  Deleted chunk {chunk_id} ZIP")


# ============================================================
# Model loaders (same as extract_full_mimic.py)
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
    return extract_batch


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
    return extract_batch


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
    return extract_batch


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
    return extract_batch


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
    return extract_batch


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
    return extract_batch


def load_clip_vit():
    """OpenAI CLIP ViT-B/16 image encoder."""
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
    return extract_batch


def load_convnextv2():
    """ConvNeXtV2-Base image encoder."""
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
    return extract_batch


def load_chess():
    import torchvision.models as models
    import torchvision.transforms as T
    model = models.resnet50(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
    non_fc_missing = [k for k in missing if not k.startswith("fc.")]
    print(f"  CheSS loaded: {len(renamed)} keys, non-FC missing: {len(non_fc_missing)}")
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().cuda()
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.5], [0.25]),
    ])
    def extract_batch(images):
        imgs = [transform(img.convert("L")) for img in images]
        batch = torch.stack(imgs).cuda()
        with torch.no_grad():
            out = model(batch).squeeze(-1).squeeze(-1)
        return out.cpu().numpy()
    return extract_batch


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
}


# ============================================================
# Core extraction from ZIP
# ============================================================

def build_zip_index(zip_path):
    """Build a lookup from image path stem to ZIP member name.

    The metadata has paths like: train/patient42142/study5/view1_frontal.jpg
    The ZIP may have:           train/patient42142/study5/view1_frontal.png
    We index by the stem (without extension) for flexible matching.
    """
    print(f"  Indexing ZIP: {os.path.basename(zip_path)}...")
    zf = zipfile.ZipFile(zip_path, "r")
    index = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        # Normalize: remove leading directory if ZIP has a root folder
        # and create a stem-based key
        stem = name.rsplit(".", 1)[0]  # remove extension
        # Try to match against metadata paths
        # Metadata: train/patientXXX/studyY/view1_frontal.jpg
        # ZIP may: train/patientXXX/studyY/view1_frontal.png
        # or: chexpert_plus/train/patientXXX/...
        index[stem] = name

        # Also try without top-level directory
        parts = name.split("/", 1)
        if len(parts) > 1:
            alt_stem = parts[1].rsplit(".", 1)[0]
            if alt_stem not in index:
                index[alt_stem] = name

    print(f"  Indexed {len(index)} files")
    return zf, index


def to_8bit_cxr(img):
    """Window a chest-X-ray PNG to 8-bit before model preprocessing.

    CheXpert Plus PNGs are 16-bit; PIL's convert('RGB'/'L') does NOT rescale a
    16-bit image to 8-bit -- it clips values >255, collapsing most of the image
    to white and destroying pathological contrast (gross anatomy survives, fine
    detail is lost). This produced embeddings that encoded demographics (sex
    AUROC ~0.94) but not findings (AUROC ~0.52). We robustly rescale any
    non-uint8 image to 8-bit via 1-99 percentile windowing; genuine 8-bit
    inputs pass through unchanged.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return img
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max()) + 1e-8
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return Image.fromarray(a.astype(np.uint8))


def extract_from_zip(model_name, zip_path, df_chunk, batch_size=32):
    """Extract embeddings for one model from one ZIP chunk.

    Returns:
        embeddings: np.array (n_extracted, embed_dim)
        image_ids: list of path_to_image strings
        n_failed: int
        failed_ids: list of (image_path, error_type) tuples
    """
    embed_dim = MODELS[model_name]["embed_dim"]

    # Load model
    print(f"\n  Loading model: {model_name}")
    extract_fn = MODEL_LOADERS[model_name]()

    # Build ZIP index
    zf, index = build_zip_index(zip_path)

    # Match metadata paths to ZIP entries
    paths = df_chunk["path_to_image"].tolist()
    matched = []
    for p in paths:
        stem = p.rsplit(".", 1)[0]  # train/patient.../view1_frontal
        if stem in index:
            matched.append((p, index[stem]))
        else:
            matched.append((p, None))

    n_found = sum(1 for _, z in matched if z is not None)
    n_missing = len(matched) - n_found
    print(f"  Matched: {n_found}/{len(matched)} ({n_missing} not in this chunk)")

    if n_found == 0:
        zf.close()
        return np.empty((0, embed_dim), dtype=np.float32), [], 0, []

    # Extract in batches
    all_emb = []
    all_ids = []
    failed = 0
    failed_ids = []
    batch_imgs = []
    batch_ids = []

    pbar = tqdm(matched, desc=f"  {model_name}")
    for img_path, zip_name in pbar:
        if zip_name is None:
            continue

        try:
            data = zf.read(zip_name)
            img = Image.open(BytesIO(data))
            img.load()
            img = to_8bit_cxr(img)  # FIX: window 16-bit PNG to 8-bit (else convert() clips >255 -> pathology lost)
            batch_imgs.append(img)
            batch_ids.append(img_path)
        except Exception as e:
            failed += 1
            failed_ids.append((img_path, f"read_error: {str(e)[:80]}"))
            continue

        if len(batch_imgs) >= batch_size:
            try:
                emb = extract_fn(batch_imgs)
                all_emb.append(emb)
                all_ids.extend(batch_ids)
            except Exception:
                # Fallback: one by one
                for single_img, single_id in zip(batch_imgs, batch_ids):
                    try:
                        emb = extract_fn([single_img])
                        all_emb.append(emb)
                        all_ids.append(single_id)
                    except Exception as e:
                        failed += 1
                        failed_ids.append((single_id, f"extract_error: {str(e)[:80]}"))
            for img in batch_imgs:
                img.close()
            batch_imgs = []
            batch_ids = []

            pbar.set_postfix(extracted=len(all_ids), failed=failed)

    # Process remaining batch
    if batch_imgs:
        try:
            emb = extract_fn(batch_imgs)
            all_emb.append(emb)
            all_ids.extend(batch_ids)
        except Exception:
            for single_img, single_id in zip(batch_imgs, batch_ids):
                try:
                    emb = extract_fn([single_img])
                    all_emb.append(emb)
                    all_ids.append(single_id)
                except Exception as e:
                    failed += 1
                    failed_ids.append((single_id, f"extract_error: {str(e)[:80]}"))
        for img in batch_imgs:
            img.close()

    zf.close()

    if all_emb:
        embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    else:
        embeddings = np.empty((0, embed_dim), dtype=np.float32)

    print(f"  {model_name}: extracted {len(all_ids)}, failed {failed}")
    return embeddings, all_ids, failed, failed_ids


# ============================================================
# Chunk-level orchestration
# ============================================================

def get_chunk_output_path(model_name, chunk_id):
    """Per-chunk embedding output path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return (
        os.path.join(OUTPUT_DIR, f"{model_name}_chunk{chunk_id}_emb.npy"),
        os.path.join(OUTPUT_DIR, f"{model_name}_chunk{chunk_id}_ids.npy"),
    )


def process_chunk(chunk_id, models, df, batch_size=32, skip_download=False):
    """Download, extract embeddings, delete for one chunk."""
    print(f"\n{'='*60}")
    print(f"  CHUNK {chunk_id}")
    print(f"{'='*60}")

    # Download
    if not skip_download:
        zip_path = download_chunk(chunk_id)
    else:
        zip_path = os.path.join(TEMP_DIR, f"png_chexpert_plus_chunk_{chunk_id}.zip")
        if not os.path.exists(zip_path):
            print(f"  ERROR: ZIP not found at {zip_path}")
            return

    # Extract embeddings for each model
    for model_name in models:
        emb_path, ids_path = get_chunk_output_path(model_name, chunk_id)

        # Skip if already done
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            existing = np.load(emb_path)
            print(f"  {model_name} chunk {chunk_id}: already done ({existing.shape[0]} images)")
            continue

        embeddings, image_ids, n_failed, failed_ids = extract_from_zip(
            model_name, zip_path, df, batch_size=batch_size
        )

        np.save(emb_path, embeddings)
        np.save(ids_path, np.array(image_ids))
        print(f"  Saved: {emb_path} ({embeddings.shape})")

        # Log failed images for reproducibility
        if failed_ids:
            fail_path = os.path.join(
                OUTPUT_DIR, f"{model_name}_chunk{chunk_id}_failed.txt"
            )
            with open(fail_path, "w") as f:
                for img_id, reason in failed_ids:
                    f.write(f"{img_id}\t{reason}\n")
            print(f"  Failed log: {fail_path} ({len(failed_ids)} entries)")

    # Delete ZIP
    if not skip_download:
        delete_chunk(chunk_id)


def merge_chunks(models):
    """Merge per-chunk embeddings into final arrays."""
    print(f"\n{'='*60}")
    print(f"  MERGING CHUNKS")
    print(f"{'='*60}")

    for model_name in models:
        all_emb = []
        all_ids = []

        for chunk_id in sorted(CHUNKS.keys()):
            emb_path, ids_path = get_chunk_output_path(model_name, chunk_id)
            if os.path.exists(emb_path):
                emb = np.load(emb_path)
                ids = np.load(ids_path, allow_pickle=True)
                all_emb.append(emb)
                all_ids.extend(ids.tolist())
                print(f"  {model_name} chunk {chunk_id}: {emb.shape[0]} images")

        if all_emb:
            merged_emb = np.concatenate(all_emb, axis=0)
            merged_ids = np.array(all_ids)

            final_emb_path = os.path.join(OUTPUT_DIR, f"{model_name}_embeddings.npy")
            final_ids_path = os.path.join(OUTPUT_DIR, f"{model_name}_image_ids.npy")
            np.save(final_emb_path, merged_emb)
            np.save(final_ids_path, merged_ids)
            print(f"  MERGED {model_name}: {merged_emb.shape}")
        else:
            print(f"  WARNING: No chunks found for {model_name}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CheXpert Plus chunk-by-chunk extraction")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all'")
    parser.add_argument("--chunk", type=int, default=None,
                        help="Process single chunk (0-4). Default: all chunks")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip AzCopy download (ZIP must be on disk)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing chunk files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Select models (supports 'all' = all loadable, or comma-separated list)
    if args.model == "all":
        models = [m for m in MODELS.keys() if m in MODEL_LOADERS]
    elif "," in args.model:
        models = [m.strip() for m in args.model.split(",") if m.strip()]
    else:
        models = [args.model]
    bad = [m for m in models if m not in MODEL_LOADERS]
    if bad:
        print(f"ERROR: no loader for {bad}. Available: {list(MODEL_LOADERS.keys())}")
        return

    print(f"Models: {models}")
    print(f"Metadata: {METADATA_CSV}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Temp: {TEMP_DIR}")
    print()

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    print(f"Total frontal images: {len(df)}")

    if args.dry_run:
        print("\nDry run - checking AzCopy...")
        result = subprocess.run([AZCOPY, "--version"], capture_output=True, text=True)
        print(f"  AzCopy: {result.stdout.strip()}")

        import shutil
        _, _, free = shutil.disk_usage(".")
        print(f"  Free disk (cwd): {free / (1024**3):.1f} GB")
        print(f"  Chunk sizes: 4x 145 GB + 1x 91 GB")
        print(f"  Need ~145 GB temp per chunk (stream from ZIP)")
        print("\nDry run complete.")
        return

    if args.merge_only:
        merge_chunks(models)
        return

    # Select chunks
    if args.chunk is not None:
        chunk_ids = [args.chunk]
    else:
        chunk_ids = sorted(CHUNKS.keys())

    # Process chunks
    start = time.time()
    for chunk_id in chunk_ids:
        process_chunk(
            chunk_id, models, df,
            batch_size=args.batch_size,
            skip_download=args.skip_download,
        )

    # Merge if all chunks done
    if args.chunk is None:
        merge_chunks(models)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
