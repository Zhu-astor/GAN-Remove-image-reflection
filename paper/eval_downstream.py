"""
CVGIP 2025 — Downstream Classification Evaluation Script
File   : eval_downstream.py
Purpose: Exhaustive evaluation of reflection-removal → YOLOv8 classification pipeline.

Tests ALL combinations of:
  - Preprocessing  : raw / Baseline-Pix2Pix-GAN / Pix2Pix+SGA-GAN
  - Test sets      : Test/Reflection (198), Test/NonReflection (198), Test/jpg (651)
  - YOLOv8 models  : CLASS_train3, CLASS_train8, DATASETS_train2, DATASETS_train3
  - Labeled val    : model.val() on config.yaml (374 imgs) and museum_data.yaml (139 imgs)
  - Paired analysis: class-agreement rate using NonReflection as pseudo-ground-truth

## Changelog
### v1.0.0 — 2026-06-06
**Added:**
- Phase 1: GAN preprocessing — saves GAN-processed images to temp directory
- Phase 2: Labeled YOLOv8 val — mAP50 / precision / recall on val splits
- Phase 3: Unlabeled predict — detection_rate, avg_confidence per (model x preprocessing x set)
- Phase 4: Paired agreement analysis — Test/Reflection vs Test/NonReflection
- Three output files: summary CSV, per-image CSV, summary TXT

==========================================================================
ENVIRONMENT
==========================================================================
Use python389 (TF 2.6.0 + ultralytics 8.2.94 + NumPy 1.19.2):
  cd "D:\\Contest\\AI GO\\paper"
  C:\\Users\\bubbl\\Desktop\\Virtualenv\\python389\\Scripts\\python.exe eval_downstream.py

DO NOT use anaconda3 base (NumPy 2.0 breaks TF 2.6).
==========================================================================
"""

import os
import sys
import csv
import shutil
import numpy as np
from pathlib import Path
from glob import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ==========================================================================
# CONFIGURATION
# ==========================================================================

ROOT      = Path(r"D:\Contest\AI GO")
PAPER_DIR = ROOT / "paper"
TEMP_DIR  = PAPER_DIR / "_downstream_temp"   # cleaned up at end of run

GAN_BASELINE_PATH = ROOT / r"GAN_Test\saved_model_12_d2\generator_300.h5"
GAN_SGA_PATH      = (ROOT / r"Model\Reflection"
                    r"\saved_model_batch8_epoch500_at_d2\generator_400.h5")

TEST_REFLECTION    = ROOT / r"GAN_Test\Dataset\Test\Reflection"
TEST_NONREFLECTION = ROOT / r"GAN_Test\Dataset\Test\NonReflection"
TEST_JPG           = ROOT / r"GAN_Test\Dataset\Test\jpg"

# model name → {path, data yaml, imgsz, nc}
YOLO_MODELS = {
    "CLASS_train3": {
        "path" : ROOT / r"Classification\runs\detect\train3\weights\best.pt",
        "data" : ROOT / r"Classification\config.yaml",
        "imgsz": 640,
        "nc"   : 8,     # other + obj1-obj7
    },
    "CLASS_train8": {
        "path" : ROOT / r"Classification\runs\detect\train8\weights\best.pt",
        "data" : ROOT / r"Classification\config.yaml",
        "imgsz": 640,
        "nc"   : 8,
    },
    "DATASETS_train2": {
        "path" : (ROOT / r"Classification\datasets"
                  r"\museum_data_annotation.v9i.yolov8"
                  r"\runs\detect\train2\weights\best.pt"),
        "data" : (ROOT / r"Classification\datasets"
                  r"\museum_data_annotation.v9i.yolov8\data.yaml"),
        "imgsz": 256,
        "nc"   : 7,     # obj1-obj7
    },
    "DATASETS_train3": {
        "path" : (ROOT / r"Classification\datasets"
                  r"\museum_data_annotation.v9i.yolov8"
                  r"\runs\detect\train3\weights\best.pt"),
        "data" : (ROOT / r"Classification\datasets"
                  r"\museum_data_annotation.v9i.yolov8\data.yaml"),
        "imgsz": 640,
        "nc"   : 7,
    },
}

IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp")
GAN_IMGSZ   = (256, 256)    # all GAN models were trained at 256×256
CONF_THRESH = 0.25          # YOLO detection confidence threshold


# ==========================================================================
# LAZY IMPORTS  (heavy frameworks loaded only when main() runs)
# ==========================================================================

_TF   = None
_YOLO = None


def _load_tf():
    global _TF
    if _TF is not None:
        return _TF
    print("Loading TensorFlow …")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        print(f"  TF {tf.__version__}  |  NumPy {np.__version__}")
        _TF = (tf, load_model)
    except Exception as exc:
        print(f"[FATAL] TensorFlow import failed: {exc}")
        print("  Use python389, not anaconda3.")
        sys.exit(1)
    return _TF


def _load_ultralytics():
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    print("Loading ultralytics …")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  ultralytics {ultralytics.__version__}")
        _YOLO = YOLO
    except Exception as exc:
        print(f"[FATAL] ultralytics import failed: {exc}")
        sys.exit(1)
    return _YOLO


# ==========================================================================
# IMAGE UTILITIES
# ==========================================================================

def _list_images(folder: Path) -> list:
    """
    Return sorted list of image paths in folder.

    Args:
        folder: directory to scan

    Returns:
        sorted list of absolute path strings
    """
    return sorted([
        p for p in glob(str(folder / "*"))
        if p.lower().endswith(IMG_EXTS)
    ])


def _load_image_for_gan(path: str) -> np.ndarray:
    """
    Load image as uint8 RGB resized to GAN_IMGSZ.

    Args:
        path: image file path

    Returns:
        np.ndarray (H, W, 3) uint8
    """
    from PIL import Image as PILImage
    return np.array(
        PILImage.open(path).convert("RGB").resize(GAN_IMGSZ, PILImage.BICUBIC),
        dtype=np.uint8,
    )


def _gan_preprocess(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [-1,1] with batch dim (1,H,W,3)."""
    return (img.astype(np.float32) / 127.5 - 1.0)[np.newaxis]


def _gan_postprocess(pred: np.ndarray) -> np.ndarray:
    """float32 [-1,1] batch output (1,H,W,3) → uint8 [0,255] (H,W,3)."""
    return np.clip((pred[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)


# ==========================================================================
# PHASE 1 — GAN preprocessing
# ==========================================================================

def phase1_gan_preprocess(gan_path: Path, src_dir: Path,
                           out_dir: Path, tag: str) -> Path:
    """
    Run a GAN generator on every image in src_dir; save results to out_dir.

    Outputs are saved as PNG with the original file stem (no extension change
    in key), so that filename-based matching works across raw/processed sets.

    Args:
        gan_path: path to Keras .h5 generator checkpoint
        src_dir : source image folder
        out_dir : destination folder (created if absent)
        tag     : display label used in progress messages

    Returns:
        out_dir after processing
    """
    from PIL import Image as PILImage
    tf, keras_load_model = _load_tf()

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _list_images(src_dir)
    n = len(paths)
    if n == 0:
        print(f"  [{tag}] WARNING: no images found in {src_dir}")
        return out_dir

    print(f"  [{tag}] {gan_path.name} on {src_dir.name}/ ({n} imgs) → {out_dir.name}/")
    model = keras_load_model(str(gan_path), compile=False)

    for i, src in enumerate(paths, 1):
        stem = Path(src).stem
        dst  = out_dir / f"{stem}.png"
        img     = _load_image_for_gan(src)
        pred    = model.predict(_gan_preprocess(img), verbose=0)
        result  = _gan_postprocess(pred)
        PILImage.fromarray(result).save(str(dst))
        if i % 100 == 0 or i == n:
            print(f"    {i}/{n}")

    del model
    tf.keras.backend.clear_session()
    return out_dir


# ==========================================================================
# PHASE 2 — Labeled YOLOv8 validation
# ==========================================================================

def phase2_labeled_val(yolo_name: str, cfg: dict) -> dict:
    """
    Run model.val() on the labeled val split for one YOLO model.

    Args:
        yolo_name: key from YOLO_MODELS
        cfg      : YOLO_MODELS[yolo_name]

    Returns:
        result dict for CSV (mAP50, precision, recall)
    """
    YOLO = _load_ultralytics()
    print(f"  [Val] {yolo_name:<22} data={cfg['data'].name}  imgsz={cfg['imgsz']}")

    model = YOLO(str(cfg["path"]))
    metrics = model.val(
        data=str(cfg["data"]),
        split="val",
        imgsz=cfg["imgsz"],
        conf=CONF_THRESH,
        save=False,
        plots=False,
        verbose=False,
    )
    mp50 = float(metrics.box.map50)
    prec = float(metrics.box.mp)
    rec  = float(metrics.box.mr)

    print(f"    mAP50={mp50:.4f}  prec={prec:.4f}  recall={rec:.4f}")
    return {
        "phase"          : "labeled_val",
        "yolo_model"     : yolo_name,
        "test_set"       : "val_split",
        "preprocessing"  : "N/A",
        "n_images"       : "",
        "mAP50"          : f"{mp50:.4f}",
        "precision"      : f"{prec:.4f}",
        "recall"         : f"{rec:.4f}",
        "detection_rate" : "",
        "avg_top1_conf"  : "",
        "class_agreement": "",
    }


# ==========================================================================
# PREDICTION HELPER
# ==========================================================================

def _predict_folder(model, folder: Path, imgsz: int, conf: float) -> list:
    """
    Run YOLOv8 predict on all images in folder.

    Uses file stem (no extension) as the identifier so that raw (.jpg) and
    GAN-processed (.png) images with the same base name can be matched.

    Args:
        model : loaded ultralytics YOLO model
        folder: image directory
        imgsz : inference resolution
        conf  : detection confidence threshold

    Returns:
        list of dicts {stem, detected, top1_class, top1_conf, n_detections}
        sorted by stem
    """
    paths = _list_images(folder)
    if not paths:
        return []

    preds = model.predict(
        source=paths,
        imgsz=imgsz,
        conf=conf,
        save=False,
        verbose=False,
        stream=False,
    )
    results = []
    for path, pred in zip(paths, preds):
        stem  = Path(path).stem
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            results.append({
                "stem"         : stem,
                "detected"     : False,
                "top1_class"   : -1,
                "top1_conf"    : 0.0,
                "n_detections" : 0,
            })
        else:
            confs    = boxes.conf.cpu().numpy()
            classes  = boxes.cls.cpu().numpy().astype(int)
            best     = int(np.argmax(confs))
            results.append({
                "stem"         : stem,
                "detected"     : True,
                "top1_class"   : int(classes[best]),
                "top1_conf"    : float(confs[best]),
                "n_detections" : len(confs),
            })
    return results


def _aggregate(per_img: list) -> dict:
    """
    Compute detection_rate and avg_top1_conf from per-image prediction list.

    Args:
        per_img: output of _predict_folder

    Returns:
        dict {n_images, detection_rate, avg_top1_conf}
    """
    n = len(per_img)
    if n == 0:
        return {"n_images": 0, "detection_rate": 0.0, "avg_top1_conf": 0.0}
    detected = [r for r in per_img if r["detected"]]
    det_rate = len(detected) / n
    avg_conf = float(np.mean([r["top1_conf"] for r in detected])) if detected else 0.0
    return {"n_images": n, "detection_rate": det_rate, "avg_top1_conf": avg_conf}


# ==========================================================================
# PHASE 3 — Unlabeled prediction sweep
# ==========================================================================

def phase3_unlabeled_predict(yolo_name: str, cfg: dict,
                              predict_combos: dict) -> tuple:
    """
    For one YOLO model, run predict on all (test_set × preprocessing) combos.

    Loads the model once, then iterates over all combinations.

    Args:
        yolo_name      : model key
        cfg            : YOLO_MODELS[yolo_name]
        predict_combos : {test_set_name: {prep_label: Path, ...}, ...}

    Returns:
        (summary_rows: list[dict], per_image_rows: list[dict])
    """
    YOLO = _load_ultralytics()
    model = YOLO(str(cfg["path"]))
    imgsz = cfg["imgsz"]

    summary_rows  = []
    per_image_rows = []

    for test_set_name, preproc_map in predict_combos.items():
        for prep_label, img_dir in preproc_map.items():
            if not img_dir.exists():
                print(f"    [SKIP] {img_dir} not found")
                continue
            paths = _list_images(img_dir)
            if not paths:
                print(f"    [SKIP] {test_set_name}/{prep_label} — no images")
                continue

            print(f"    {yolo_name:<22} {prep_label:<16} {test_set_name} ({len(paths)} imgs)")
            per_img = _predict_folder(model, img_dir, imgsz, CONF_THRESH)
            agg     = _aggregate(per_img)

            summary_rows.append({
                "phase"          : "unlabeled_predict",
                "yolo_model"     : yolo_name,
                "test_set"       : test_set_name,
                "preprocessing"  : prep_label,
                "n_images"       : agg["n_images"],
                "mAP50"          : "",
                "precision"      : "",
                "recall"         : "",
                "detection_rate" : f"{agg['detection_rate']:.4f}",
                "avg_top1_conf"  : f"{agg['avg_top1_conf']:.4f}",
                "class_agreement": "",
            })

            for r in per_img:
                per_image_rows.append({
                    "yolo_model"   : yolo_name,
                    "test_set"     : test_set_name,
                    "preprocessing": prep_label,
                    "stem"         : r["stem"],
                    "detected"     : r["detected"],
                    "top1_class"   : r["top1_class"],
                    "top1_conf"    : f"{r['top1_conf']:.4f}",
                    "n_detections" : r["n_detections"],
                })

    del model
    return summary_rows, per_image_rows


# ==========================================================================
# PHASE 4 — Paired agreement analysis
# ==========================================================================

def phase4_paired_agreement(yolo_name: str, cfg: dict,
                             refl_dirs: dict,
                             nonrefl_dir: Path) -> list:
    """
    Compute class-agreement rate for reflection images vs clean images.

    NonReflection predictions serve as pseudo-ground-truth.  Agreement rate
    represents how often the model predicts the same class on a processed
    reflection image as it does on the clean counterpart.

    Args:
        yolo_name   : model key
        cfg         : YOLO_MODELS[yolo_name]
        refl_dirs   : {preprocessing_label: Path} for reflection variants
        nonrefl_dir : Test/NonReflection folder

    Returns:
        list of result dicts (one per preprocessing condition)
    """
    YOLO  = _load_ultralytics()
    model = YOLO(str(cfg["path"]))
    imgsz = cfg["imgsz"]

    print(f"\n  [Paired] {yolo_name}")
    nonrefl_preds = _predict_folder(model, nonrefl_dir, imgsz, CONF_THRESH)
    # key = stem (no extension), value = predicted class (-1 = no detection)
    pseudo_gt = {r["stem"]: r["top1_class"] for r in nonrefl_preds}

    rows = []
    for prep_label, refl_dir in refl_dirs.items():
        if not refl_dir.exists() or not _list_images(refl_dir):
            print(f"    [SKIP] {prep_label}: {refl_dir} empty or missing")
            continue

        preds   = _predict_folder(model, refl_dir, imgsz, CONF_THRESH)
        n_both  = 0
        n_agree = 0

        for r in preds:
            gt_cls = pseudo_gt.get(r["stem"])
            if gt_cls is None:
                # Image has no NonReflection counterpart — skip
                continue
            if gt_cls == -1:
                # Model failed on NonReflection image too — skip (not a fair test)
                continue
            n_both += 1
            if r["top1_class"] == gt_cls:
                n_agree += 1

        agree_rate = n_agree / n_both if n_both > 0 else 0.0
        print(f"    {prep_label:<16}: {n_agree}/{n_both} = {agree_rate:.4f}")

        rows.append({
            "phase"          : "paired_agreement",
            "yolo_model"     : yolo_name,
            "test_set"       : "Reflection_vs_NonReflection",
            "preprocessing"  : prep_label,
            "n_images"       : n_both,
            "mAP50"          : "",
            "precision"      : "",
            "recall"         : "",
            "detection_rate" : "",
            "avg_top1_conf"  : "",
            "class_agreement": f"{agree_rate:.4f}",
        })

    del model
    return rows


# ==========================================================================
# OUTPUT
# ==========================================================================

_CSV_FIELDS = [
    "phase", "yolo_model", "test_set", "preprocessing",
    "n_images", "mAP50", "precision", "recall",
    "detection_rate", "avg_top1_conf", "class_agreement",
]

_PER_IMAGE_FIELDS = [
    "yolo_model", "test_set", "preprocessing",
    "stem", "detected", "top1_class", "top1_conf", "n_detections",
]


def _write_csv(rows: list, path: Path, fields: list) -> None:
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[Saved] {path}")


def _write_summary(all_rows: list, path: Path) -> None:
    lines = [
        "CVGIP 2025 — Downstream Classification Evaluation Summary",
        "=" * 72,
        "",
        "PHASE 2: Labeled Val (mAP50 / Precision / Recall)",
        "-" * 72,
        f"{'Model':<22} {'mAP50':>7} {'Prec':>7} {'Recall':>7}",
        "-" * 72,
    ]
    for r in all_rows:
        if r["phase"] != "labeled_val":
            continue
        lines.append(f"{r['yolo_model']:<22} {r['mAP50']:>7} "
                     f"{r['precision']:>7} {r['recall']:>7}")

    lines += [
        "",
        "PHASE 3: Unlabeled Prediction",
        "-" * 72,
        f"{'Model':<22} {'TestSet':<22} {'Preprocessing':<16} {'DetRate':>8} {'AvgConf':>8}",
        "-" * 72,
    ]
    for r in all_rows:
        if r["phase"] != "unlabeled_predict":
            continue
        lines.append(
            f"{r['yolo_model']:<22} {r['test_set']:<22} "
            f"{r['preprocessing']:<16} "
            f"{r['detection_rate']:>8} {r['avg_top1_conf']:>8}"
        )

    lines += [
        "",
        "PHASE 4: Paired Class Agreement  (pseudo-GT = NonReflection prediction)",
        "-" * 72,
        f"{'Model':<22} {'Preprocessing':<16} {'Agreement':>10} {'N':>6}",
        "-" * 72,
    ]
    for r in all_rows:
        if r["phase"] != "paired_agreement":
            continue
        lines.append(
            f"{r['yolo_model']:<22} {r['preprocessing']:<16}"
            f" {r['class_agreement']:>10} {str(r['n_images']):>6}"
        )

    lines += [
        "",
        "=" * 72,
        "NOTES FOR TABLE 2 (MUST-3):",
        "  Use Phase 4 paired_agreement rows.",
        "  'raw'         ≈ accuracy on reflection images (target: 92.7%)",
        "  'SGA_GAN'     ≈ accuracy after SGA removal    (target: 94.5%)",
        "  'Baseline_GAN'≈ accuracy after Baseline removal",
        "  agreement_rate = (# images where predicted class matches NonReflection) / N",
        "",
        "  Best YOLO models: CLASS_train3 and CLASS_train8 (both mAP50=0.995 on val).",
    ]

    with open(str(path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Saved] {path}")


# ==========================================================================
# PATH VALIDATION
# ==========================================================================

def validate_paths() -> bool:
    """Check all required paths before running the (slow) evaluation."""
    required = {
        "GAN Baseline"      : GAN_BASELINE_PATH,
        "GAN SGA"           : GAN_SGA_PATH,
        "Test/Reflection"   : TEST_REFLECTION,
        "Test/NonReflection": TEST_NONREFLECTION,
        "Test/jpg"          : TEST_JPG,
    }
    for name, cfg in YOLO_MODELS.items():
        required[f"YOLO {name}"]  = cfg["path"]
        required[f"data  {name}"] = cfg["data"]

    ok = True
    for label, path in required.items():
        exists = path.exists()
        mark = "OK" if exists else "MISSING"
        print(f"  [{mark:^7}] {label}: {path.name}")
        if not exists:
            ok = False
    return ok


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print("\n" + "=" * 72)
    print("  CVGIP 2025 — eval_downstream.py  v1.0.0")
    print("=" * 72)

    # ── 0. Validate paths ───────────────────────────────────────────────────
    print("\n[0] Validating paths")
    if not validate_paths():
        print("\n[ABORT] Fix the MISSING paths above before re-running.")
        sys.exit(1)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Temp directories for GAN-processed images
    temp_baseline_refl = TEMP_DIR / "baseline_reflection"
    temp_sga_refl      = TEMP_DIR / "sga_reflection"
    temp_baseline_jpg  = TEMP_DIR / "baseline_jpg"
    temp_sga_jpg       = TEMP_DIR / "sga_jpg"

    all_rows      = []
    per_image_rows = []

    # ── 1. GAN preprocessing ────────────────────────────────────────────────
    print("\n[1] GAN preprocessing")
    for (gan_path, src, dst, tag) in [
        (GAN_BASELINE_PATH, TEST_REFLECTION, temp_baseline_refl, "Baseline/Refl"),
        (GAN_SGA_PATH,      TEST_REFLECTION, temp_sga_refl,      "SGA/Refl"),
        (GAN_BASELINE_PATH, TEST_JPG,        temp_baseline_jpg,  "Baseline/jpg"),
        (GAN_SGA_PATH,      TEST_JPG,        temp_sga_jpg,       "SGA/jpg"),
    ]:
        try:
            phase1_gan_preprocess(gan_path, src, dst, tag)
        except Exception as exc:
            print(f"  [ERROR] GAN {tag}: {exc}")

    # ── 2. Labeled val ──────────────────────────────────────────────────────
    print("\n[2] Labeled YOLOv8 validation")
    for yolo_name, cfg in YOLO_MODELS.items():
        try:
            all_rows.append(phase2_labeled_val(yolo_name, cfg))
        except Exception as exc:
            print(f"  [ERROR] val {yolo_name}: {exc}")

    # ── 3. Unlabeled prediction sweep ───────────────────────────────────────
    print("\n[3] Unlabeled prediction sweep")
    predict_combos = {
        "Test_Reflection": {
            "raw"         : TEST_REFLECTION,
            "Baseline_GAN": temp_baseline_refl,
            "SGA_GAN"     : temp_sga_refl,
        },
        "Test_jpg": {
            "raw"         : TEST_JPG,
            "Baseline_GAN": temp_baseline_jpg,
            "SGA_GAN"     : temp_sga_jpg,
        },
        "Test_NonReflection": {
            "raw": TEST_NONREFLECTION,
        },
    }
    for yolo_name, cfg in YOLO_MODELS.items():
        try:
            s_rows, pi_rows = phase3_unlabeled_predict(
                yolo_name, cfg, predict_combos
            )
            all_rows.extend(s_rows)
            per_image_rows.extend(pi_rows)
        except Exception as exc:
            print(f"  [ERROR] predict sweep {yolo_name}: {exc}")

    # ── 4. Paired agreement ──────────────────────────────────────────────────
    print("\n[4] Paired class-agreement analysis")
    refl_dirs = {
        "raw"         : TEST_REFLECTION,
        "Baseline_GAN": temp_baseline_refl,
        "SGA_GAN"     : temp_sga_refl,
    }
    for yolo_name, cfg in YOLO_MODELS.items():
        try:
            rows = phase4_paired_agreement(
                yolo_name, cfg, refl_dirs, TEST_NONREFLECTION
            )
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  [ERROR] paired {yolo_name}: {exc}")

    # ── 5. Save outputs ──────────────────────────────────────────────────────
    print("\n[5] Saving results")
    _write_csv(all_rows,       PAPER_DIR / "eval_downstream_results.csv",  _CSV_FIELDS)
    _write_csv(per_image_rows, PAPER_DIR / "eval_downstream_per_image.csv", _PER_IMAGE_FIELDS)
    _write_summary(all_rows,   PAPER_DIR / "eval_downstream_summary.txt")

    # ── 6. Cleanup ───────────────────────────────────────────────────────────
    print(f"\n[6] Removing temp dir: {TEMP_DIR}")
    shutil.rmtree(str(TEMP_DIR), ignore_errors=True)

    # ── Console summary (key rows for paper) ────────────────────────────────
    print("\n" + "=" * 72)
    print("  KEY RESULTS — Paired Agreement (for Table 2 / MUST-3)")
    print("=" * 72)
    agree_rows = [r for r in all_rows if r["phase"] == "paired_agreement"]
    if agree_rows:
        print(f"  {'Model':<22} {'Preprocessing':<16} {'Agreement':>10} {'N':>6}")
        print("  " + "-" * 58)
        for r in agree_rows:
            print(f"  {r['yolo_model']:<22} {r['preprocessing']:<16}"
                  f" {r['class_agreement']:>10} {str(r['n_images']):>6}")
    else:
        print("  (no data — Test/Reflection may be empty)")

    print("\n  Labeled Val mAP50:")
    for r in all_rows:
        if r["phase"] == "labeled_val":
            print(f"  {r['yolo_model']:<22}  mAP50={r['mAP50']}")

    print("\nDone. Copy agreement rates into Table 2 of cvgip2025_chinese.py.")


if __name__ == "__main__":
    main()
