"""
CVGIP 2025 — SIRR Quantitative Evaluation Script
File   : eval_metrics.py
Purpose: Compute PSNR / SSIM / LPIPS on the public SIRR test set for
         Baseline Pix2Pix and Pix2Pix+SGA models.
Output :
    D:/Contest/AI GO/paper/eval_results.csv   — per-image breakdown
    D:/Contest/AI GO/paper/eval_summary.txt   — table-ready numbers

==========================================================================
ENVIRONMENT SETUP
==========================================================================
This project was trained with the virtualenv at:
  C:/Users/bubbl/Desktop/Virtualenv/python389/
  Python 3.8.9 | TensorFlow 2.6.0 | NumPy 1.19.2 | keras_contrib OK

DO NOT use anaconda3 base (NumPy 2.0 breaks TF 2.18).

Step 1 - Install lpips (one-time, torch 1.7.1 is already present):
  python389/Scripts/pip.exe install lpips

Step 2 - Run this script:
  python389/Scripts/python.exe eval_metrics.py
==========================================================================
"""

import os
import sys
import csv
import numpy as np
from glob import glob
from PIL import Image

# Suppress TF boot noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Loading TensorFlow...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    print(f"  TF {tf.__version__}  |  NumPy {np.__version__}")
except Exception as e:
    print(f"[FATAL] Cannot import TensorFlow: {e}")
    print("  Run: pip install \"numpy<2\"  then retry.")
    sys.exit(1)

# LPIPS is optional; we skip it gracefully if torch/lpips is missing
LPIPS_AVAILABLE = False
lpips_fn = None
try:
    import torch
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net='alex', verbose=False)
    if torch.cuda.is_available():
        lpips_fn = lpips_fn.cuda()
    LPIPS_AVAILABLE = True
    print("  lpips loaded (AlexNet backend)")
except ImportError:
    print("  [INFO] lpips / torch not found — LPIPS will be skipped.")
    print("         Install: pip install lpips torch torchvision")

# ==========================================================================
# PATHS — adjust if your dataset location differs
# ==========================================================================
REFLECTION_DIR = r"D:\Contest\AI GO\github\Dataset2\Test\Reflection"
GT_DIR         = r"D:\Contest\AI GO\github\Dataset2\Test\NonReflection"
OUTPUT_DIR     = r"D:\Contest\AI GO\paper"

# SGA model: batch8, epoch500, evaluated at epoch 400
SGA_MODEL_PATH = (
    r"D:\Contest\AI GO\Model\Reflection"
    r"\saved_model_batch8_epoch500_at_d2\generator_400.h5"
)
# Baseline Pix2Pix: no attention module, best available checkpoint is epoch 300
BASELINE_MODEL_PATH = (
    r"D:\Contest\AI GO\GAN_Test"
    r"\saved_model_12_d2\generator_300.h5"
)

# Both saved models were trained at 256x256 (confirmed by model.input_shape).
# The paper §3.6 says 512x512 — that needs to be corrected to 256x256.
IMG_SIZE = (256, 256)


# ==========================================================================
# IMAGE UTILITIES
# ==========================================================================

def load_image(path: str, size: tuple = IMG_SIZE) -> np.ndarray:
    """Load image as uint8 numpy array (H, W, 3)."""
    img = Image.open(path).convert('RGB').resize(size, Image.BICUBIC)
    return np.array(img, dtype=np.uint8)


def preprocess(img_np: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [-1,1] with batch dimension (1, H, W, 3)."""
    return (img_np.astype(np.float32) / 127.5 - 1.0)[np.newaxis]


def postprocess(pred: np.ndarray) -> np.ndarray:
    """Model output [-1,1] → uint8 [0,255] without batch dimension (H, W, 3)."""
    out = (pred[0] + 1.0) * 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================================
# METRIC FUNCTIONS
# ==========================================================================

def compute_psnr_ssim(pred: np.ndarray, gt: np.ndarray):
    """
    Compute PSNR and SSIM via tf.image (no skimage/scipy dependency).
    Args:
        pred: uint8 (H, W, 3)
        gt  : uint8 (H, W, 3)
    Returns:
        (psnr: float, ssim: float)
    """
    pred_t = tf.convert_to_tensor(pred[np.newaxis].astype(np.float32))
    gt_t   = tf.convert_to_tensor(gt[np.newaxis].astype(np.float32))
    psnr = float(tf.image.psnr(pred_t, gt_t, max_val=255.0).numpy()[0])
    ssim = float(tf.image.ssim(pred_t, gt_t, max_val=255.0).numpy()[0])
    return psnr, ssim


def compute_lpips(pred: np.ndarray, gt: np.ndarray):
    """
    Compute LPIPS using AlexNet backend.
    Args:
        pred: uint8 (H, W, 3)
        gt  : uint8 (H, W, 3)
    Returns:
        float | None (if lpips not available)
    """
    if not LPIPS_AVAILABLE:
        return None

    import torch  # already imported above but explicit for clarity

    # lpips expects float tensor in [-1, 1], shape (1, 3, H, W)
    def to_lpips_tensor(img_uint8):
        t = torch.from_numpy(img_uint8).permute(2, 0, 1).unsqueeze(0).float()
        return t / 127.5 - 1.0  # normalize to [-1, 1]

    pred_t = to_lpips_tensor(pred)
    gt_t   = to_lpips_tensor(gt)
    if torch.cuda.is_available():
        pred_t, gt_t = pred_t.cuda(), gt_t.cuda()

    with torch.no_grad():
        return float(lpips_fn(pred_t, gt_t).item())


# ==========================================================================
# EVALUATION LOOP
# ==========================================================================

def evaluate_model(model, reflection_paths: list, gt_dir: str,
                   model_name: str) -> list:
    """
    Run inference + metrics for every test pair.

    Args:
        model            : loaded Keras model
        reflection_paths : sorted list of reflection image paths
        gt_dir           : directory with ground-truth (NonReflection) images
        model_name       : display name for progress output

    Returns:
        list of dicts with keys: model, file, psnr, ssim, lpips
    """
    results = []
    n = len(reflection_paths)
    skipped = 0
    print(f"\n{'='*60}")
    print(f"  Model : {model_name}")
    print(f"  Pairs : {n}  |  Resolution: {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"{'='*60}")

    for i, ref_path in enumerate(reflection_paths, 1):
        fname   = os.path.basename(ref_path)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            print(f"  [SKIP] GT missing: {fname}")
            skipped += 1
            continue

        ref_np = load_image(ref_path)
        gt_np  = load_image(gt_path)

        pred_raw = model.predict(preprocess(ref_np), verbose=0)
        pred_np  = postprocess(pred_raw)

        psnr, ssim = compute_psnr_ssim(pred_np, gt_np)
        lpips_val  = compute_lpips(pred_np, gt_np)

        results.append({
            'model': model_name,
            'file' : fname,
            'psnr' : psnr,
            'ssim' : ssim,
            'lpips': lpips_val if lpips_val is not None else '',
        })

        # Progress update every 50 images
        if i % 50 == 0 or i == n:
            valid = results  # all so far
            avg_psnr = np.mean([r['psnr'] for r in valid])
            avg_ssim = np.mean([r['ssim'] for r in valid])
            lpips_so_far = [r['lpips'] for r in valid if r['lpips'] != '']
            avg_lpips_str = (f"{np.mean(lpips_so_far):.4f}"
                             if lpips_so_far else "N/A")
            print(f"  [{i:>3}/{n}]  PSNR {avg_psnr:.3f}  "
                  f"SSIM {avg_ssim:.4f}  LPIPS {avg_lpips_str}")

    if skipped:
        print(f"  [WARNING] {skipped} pairs skipped (GT not found)")

    return results


# ==========================================================================
# OUTPUT FORMATTING
# ==========================================================================

def print_and_collect_summary(results_by_model: dict) -> list:
    """Print final table and return rows for file saving."""
    header  = f"\n{'Model':<32} {'N':>5} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8}"
    divider = "─" * 65
    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print(header)
    print(divider)

    summary_rows = []
    for model_name, results in results_by_model.items():
        if not results:
            continue
        psnr_vals  = [r['psnr'] for r in results]
        ssim_vals  = [r['ssim'] for r in results]
        lpips_vals = [r['lpips'] for r in results if r['lpips'] != '']

        mean_psnr  = np.mean(psnr_vals)
        mean_ssim  = np.mean(ssim_vals)
        mean_lpips = np.mean(lpips_vals) if lpips_vals else float('nan')
        lpips_str  = f"{mean_lpips:.4f}" if not np.isnan(mean_lpips) else "N/A"

        print(f"  {model_name:<30} {len(results):>5} "
              f"{mean_psnr:>8.3f} {mean_ssim:>8.4f} {lpips_str:>8}")

        summary_rows.append({
            'model' : model_name,
            'n'     : len(results),
            'psnr'  : f"{mean_psnr:.3f}",
            'ssim'  : f"{mean_ssim:.4f}",
            'lpips' : lpips_str,
        })

    print("=" * 65)
    return summary_rows


def save_per_image_csv(results_by_model: dict, output_dir: str) -> str:
    """Save per-image breakdown as CSV."""
    csv_path = os.path.join(output_dir, 'eval_results.csv')
    all_rows = []
    for results in results_by_model.values():
        all_rows.extend(results)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['model', 'file', 'psnr', 'ssim', 'lpips']
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[Saved] Per-image CSV  → {csv_path}")
    return csv_path


def save_summary_txt(summary_rows: list, n_test: int, output_dir: str) -> str:
    """Save table-ready summary for paper copy-paste."""
    txt_path = os.path.join(output_dir, 'eval_summary.txt')
    lines = [
        "CVGIP 2025 — SIRR Quantitative Evaluation",
        f"Test set : Dataset2/Test  ({n_test} pairs total scanned)",
        f"Resolution: {IMG_SIZE[0]}×{IMG_SIZE[1]}",
        "Metrics  : PSNR (dB) ↑  |  SSIM ↑  |  LPIPS ↓ (AlexNet)",
        "=" * 65,
        f"{'Model':<32} {'N':>5} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}",
        "─" * 65,
    ]
    for row in summary_rows:
        lines.append(
            f"{row['model']:<32} {row['n']:>5} "
            f"{row['psnr']:>8} {row['ssim']:>8} {row['lpips']:>8}"
        )
    lines.append("=" * 65)
    lines.append("")
    lines.append("Copy the numbers above into Table 1 of cvgip2025_chinese.py:")
    lines.append("  MUST-1 = Baseline Pix2Pix  PSNR / SSIM / LPIPS")
    lines.append("  MUST-2 = Pix2Pix+SGA       PSNR / SSIM / LPIPS")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"[Saved] Summary TXT    → {txt_path}")
    return txt_path


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    # Validate paths
    for label, path in [
        ("Reflection test dir", REFLECTION_DIR),
        ("Ground-truth dir",    GT_DIR),
        ("SGA model",           SGA_MODEL_PATH),
        ("Baseline model",      BASELINE_MODEL_PATH),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found:\n  {path}")
            sys.exit(1)

    # Collect test images
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    reflection_paths = sorted([
        p for p in glob(os.path.join(REFLECTION_DIR, '*'))
        if p.lower().endswith(exts)
    ])
    n_total = len(reflection_paths)
    print(f"\nTest images found : {n_total}")

    if n_total == 0:
        print("[ERROR] No images found in REFLECTION_DIR.")
        sys.exit(1)

    results_by_model = {}

    # --- Evaluate SGA model ---
    print(f"\nLoading SGA model...")
    print(f"  {SGA_MODEL_PATH}")
    sga_model = keras_load_model(SGA_MODEL_PATH, compile=False)
    results_by_model['Pix2Pix+SGA (Ours)'] = evaluate_model(
        sga_model, reflection_paths, GT_DIR, 'Pix2Pix+SGA (Ours)'
    )
    # Free VRAM before loading the next model
    del sga_model
    tf.keras.backend.clear_session()

    # --- Evaluate Baseline model ---
    print(f"\nLoading Baseline Pix2Pix model...")
    print(f"  {BASELINE_MODEL_PATH}")
    baseline_model = keras_load_model(BASELINE_MODEL_PATH, compile=False)
    results_by_model['Baseline Pix2Pix'] = evaluate_model(
        baseline_model, reflection_paths, GT_DIR, 'Baseline Pix2Pix'
    )
    del baseline_model
    tf.keras.backend.clear_session()

    # --- Output ---
    summary_rows = print_and_collect_summary(results_by_model)
    save_per_image_csv(results_by_model, OUTPUT_DIR)
    save_summary_txt(summary_rows, n_total, OUTPUT_DIR)

    print("\nDone. Paste the numbers from eval_summary.txt into")
    print("  cvgip2025_chinese.py  →  表1 (MUST-1, MUST-2)")


if __name__ == '__main__':
    main()
