"""
eval_sga_all.py
Evaluate every SGA checkpoint and report PSNR / SSIM / LPIPS per epoch.
Run:
  python389/Scripts/python.exe "D:/Contest/AI GO/paper/eval_sga_all.py"
"""
import os, warnings, csv
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
from glob import glob
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

import torch, lpips as lpips_lib
lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False)  # CPU only (torch 1.7 + sm_89 incompatible)

REFLECTION_DIR = r"D:\Contest\AI GO\github\Dataset2\Test\Reflection"
GT_DIR         = r"D:\Contest\AI GO\github\Dataset2\Test\NonReflection"
OUTPUT_DIR     = r"D:\Contest\AI GO\paper"

# All SGA checkpoints to evaluate
SGA_CHECKPOINTS = [
    # (label, path, img_size)
    ("SGA-256 ep100", r"D:\Contest\AI GO\GAN_Test\saved_model_14_at_d2\generator_100.h5", 256),
    ("SGA-256 ep200", r"D:\Contest\AI GO\GAN_Test\saved_model_14_at_d2\generator_200.h5", 256),
    ("SGA-256 ep300", r"D:\Contest\AI GO\GAN_Test\saved_model_14_at_d2\generator_300.h5", 256),
    ("SGA-256 ep400", r"D:\Contest\AI GO\GAN_Test\saved_model_14_at_d2\generator_400.h5", 256),
    ("SGA-256 ep500", r"D:\Contest\AI GO\GAN_Test\saved_model_14_at_d2\generator_500.h5", 256),
    ("SGA-512 ep360", r"D:\Contest\AI GO\GAN_Test\saved_model_15_at_d2_512_batch4\generator_360.h5", 512),
]


def load_image(path, size):
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)

def preprocess(img):
    return (img.astype(np.float32) / 127.5 - 1.0)[np.newaxis]

def postprocess(pred):
    return np.clip((pred[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)

def psnr_ssim(pred, gt):
    p = tf.convert_to_tensor(pred[np.newaxis].astype(np.float32))
    g = tf.convert_to_tensor(gt[np.newaxis].astype(np.float32))
    return (float(tf.image.psnr(p, g, max_val=255.0).numpy()[0]),
            float(tf.image.ssim(p, g, max_val=255.0).numpy()[0]))

def compute_lpips(pred, gt):
    def to_t(x):
        return torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
    with torch.no_grad():
        return float(lpips_fn(to_t(pred), to_t(gt)).item())


def evaluate_checkpoint(label, h5path, img_size):
    exts = (".jpg", ".jpeg", ".png")
    ref_paths = sorted([p for p in glob(os.path.join(REFLECTION_DIR, "*")) if p.lower().endswith(exts)])
    n = len(ref_paths)

    model = load_model(h5path, compile=False)
    psnr_list, ssim_list, lpips_list = [], [], []

    for i, ref_path in enumerate(ref_paths, 1):
        fname = os.path.basename(ref_path)
        gt_path = os.path.join(GT_DIR, fname)
        if not os.path.exists(gt_path):
            continue

        ref_np = load_image(ref_path, img_size)
        gt_np  = load_image(gt_path, img_size)
        pred   = postprocess(model.predict(preprocess(ref_np), verbose=0))

        ps, ss = psnr_ssim(pred, gt_np)
        lp     = compute_lpips(pred, gt_np)
        psnr_list.append(ps); ssim_list.append(ss); lpips_list.append(lp)

        if i % 100 == 0 or i == n:
            print(f"  [{i:>3}/{n}]  PSNR {np.mean(psnr_list):.3f}  "
                  f"SSIM {np.mean(ssim_list):.4f}  LPIPS {np.mean(lpips_list):.4f}")

    del model
    tf.keras.backend.clear_session()

    return {
        "label": label, "n": len(psnr_list),
        "psnr":  np.mean(psnr_list),
        "ssim":  np.mean(ssim_list),
        "lpips": np.mean(lpips_list),
    }


def main():
    results = []
    for label, path, size in SGA_CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"  {label}  ({size}px)  {os.path.basename(path)}")
        print(f"{'='*60}")
        r = evaluate_checkpoint(label, path, size)
        results.append(r)
        print(f"  >> PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}  LPIPS {r['lpips']:.4f}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  ALL SGA CHECKPOINTS SUMMARY")
    print(f"  {'Label':<18} {'N':>4} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"  {'-'*60}")
    best_psnr = max(results, key=lambda x: x["psnr"])
    for r in results:
        marker = " <-- best PSNR" if r is best_psnr else ""
        print(f"  {r['label']:<18} {r['n']:>4} {r['psnr']:>8.3f} {r['ssim']:>8.4f} {r['lpips']:>8.4f}{marker}")
    print(f"{'='*70}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "eval_sga_epochs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label","n","psnr","ssim","lpips"])
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n[Saved] {csv_path}")


if __name__ == "__main__":
    main()
