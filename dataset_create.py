"""
Synthetic dataset generation: composite one or more crop PNGs onto backgrounds and write YOLO labels.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# Keep literals in sync with main.FORMAT_*
FORMAT_YOLO = "YOLO bounding box"
FORMAT_YOLO_OBB = "YOLO oriented bounding box"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images_in_dir(directory: Path) -> list[Path]:
    """List all images in a directory."""
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(f)
    return out


def list_images_in_tree(root: Path) -> list[Path]:
    """All supported image files under ``root``, including nested subfolders."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return out


def load_image_bgra(path: str | Path) -> np.ndarray | None:
    """BGR or BGRA; if BGR, alpha is set to 255."""
    p = str(path)
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        img[:, :, 3] = 255
    elif img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        img = cv2.merge([b, g, r, a])
    elif img.shape[2] == 4:
        pass
    else:
        return None
    return img


def load_image_bgr(path: str | Path) -> np.ndarray | None:
    """Load a BGR image from disk."""
    p = str(path)
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def balanced_crop_indices(num_samples: int, num_crops: int, rng: random.Random) -> list[int]:
    """Each crop index appears floor or ceil(num_samples/num_crops) times; list is shuffled."""
    if num_crops <= 0 or num_samples <= 0:
        return []
    base = num_samples // num_crops
    rem = num_samples % num_crops
    out: list[int] = []
    for i in range(num_crops):
        cnt = base + (1 if i < rem else 0)
        out.extend([i] * cnt)
    rng.shuffle(out)
    return out


PLACE_K_IOU_MAX_ATTEMPTS = 400


def scaled_crop_size(
    crop_h: int,
    crop_w: int,
    bg_h: int,
    bg_w: int,
    scale_pct: int,
) -> tuple[int, int] | None:
    """
    Same scaling rule as ``scale_and_place_rect``, without choosing a position.
    Returns (new_w, new_h) or None if impossible.
    """
    s = max(10, min(90, scale_pct)) / 100.0
    max_w = max(1, int(bg_w * s))
    max_h = max(1, int(bg_h * s))
    if crop_w < 1 or crop_h < 1:
        return None
    scale = min(max_w / float(crop_w), max_h / float(crop_h))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    new_w = min(new_w, bg_w)
    new_h = min(new_h, bg_h)
    if new_w < 1 or new_h < 1 or new_w > bg_w or new_h > bg_h:
        return None
    return new_w, new_h


def scale_and_place_rect(
    crop_h: int,
    crop_w: int,
    bg_h: int,
    bg_w: int,
    scale_pct: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    """
    Uniform scale so the crop fits inside a max box of
    (bg_w * scale_pct/100) x (bg_h * scale_pct/100), preserving aspect ratio.
    ``bg_w``/``bg_h`` are the output image size (background already resized to resolution).
    Returns (px, py, new_w, new_h) top-left and size, or None if impossible.
    """
    r = scaled_crop_size(crop_h, crop_w, bg_h, bg_w, scale_pct)
    if r is None:
        return None
    new_w, new_h = r
    px = rng.randint(0, bg_w - new_w)
    py = rng.randint(0, bg_h - new_h)
    return px, py, new_w, new_h


def axis_aligned_iou_xywh(
    ax: int, ay: int, aw: int, ah: int, bx: int, by: int, bw: int, bh: int
) -> float:
    """IoU of two axis-aligned integer rectangles in pixel space (xywh, top-left, size)."""
    if aw < 1 or ah < 1 or bw < 1 or bh < 1:
        return 0.0
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    a1, a2 = float(aw * ah), float(bw * bh)
    union = a1 + a2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def try_place_k_rects_iou(
    box_sizes: list[tuple[int, int]],
    bg_w: int,
    bg_h: int,
    max_iou: float,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    """
    Random top-left for each (w, h) in ``box_sizes`` so that every pair has IoU <= ``max_iou``.
    100% max overlap (``max_iou`` >= 1) skips the pairwise check. Returns list of (px, py) or None.
    """
    k = len(box_sizes)
    if k == 0:
        return []
    for nw, nh in box_sizes:
        if nw < 1 or nh < 1 or nw > bg_w or nh > bg_h:
            return None
    for _ in range(PLACE_K_IOU_MAX_ATTEMPTS):
        pos_wh: list[tuple[int, int, int, int]] = []
        for nw, nh in box_sizes:
            px = rng.randint(0, bg_w - nw)
            py = rng.randint(0, bg_h - nh)
            pos_wh.append((px, py, nw, nh))
        if max_iou >= 0.999999:
            return [(p[0], p[1]) for p in pos_wh]
        valid = True
        for i in range(k):
            for j in range(i + 1, k):
                a, b = pos_wh[i], pos_wh[j]
                if axis_aligned_iou_xywh(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]) > max_iou + 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return [(p[0], p[1]) for p in pos_wh]
    return None


def resize_bgra(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Resize a BGRA image to the given width and height."""
    if new_w < 1 or new_h < 1:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def alpha_blend_roi(canvas_bgr: np.ndarray, patch_bgra: np.ndarray, px: int, py: int) -> None:
    """Blend patch_bgra onto canvas_bgr at (px, py); modifies canvas in place."""
    ph, pw = patch_bgra.shape[:2]
    h, w = canvas_bgr.shape[:2]
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(w, px + pw), min(h, py + ph)
    if x0 >= x1 or y0 >= y1:
        return
    psx0 = x0 - px
    psy0 = y0 - py
    psx1 = psx0 + (x1 - x0)
    psy1 = psy0 + (y1 - y0)
    roi = canvas_bgr[y0:y1, x0:x1]
    sub = patch_bgra[psy0:psy1, psx0:psx1]
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    bgr = sub[:, :, :3].astype(np.float32)
    bg = roi.astype(np.float32)
    blended = bgr * alpha + bg * (1.0 - alpha)
    roi[:] = np.clip(blended, 0, 255).astype(np.uint8)


def rect_corners_xyxy(px: float, py: float, w: float, h: float) -> tuple[tuple[float, float], ...]:
    """Top-left, top-right, bottom-right, bottom-left (clockwise), float pixel coords."""
    x0, y0 = float(px), float(py)
    x1, y1 = float(px + w), float(py)
    x2, y2 = float(px + w), float(py + h)
    x3, y3 = float(px), float(py + h)
    return (x0, y0), (x1, y1), (x2, y2), (x3, y3)


def yolo_obb_line(class_id: int, corners: tuple[tuple[float, float], ...], img_w: int, img_h: int) -> str:
    """Ultralytics-style normalized polygon (x y) x4, clockwise from TL."""
    parts = [str(class_id)]
    ih, iw = float(img_h), float(img_w)
    for x, y in corners:
        parts.append(f"{x / iw:.6f}")
        parts.append(f"{y / ih:.6f}")
    return " ".join(parts)


def yolo_bbox_line(
    class_id: int, px: float, py: float, w: float, h: float, img_w: int, img_h: int
) -> str:
    """YOLO bounding box line."""
    xc = (px + w / 2.0) / img_w
    yc = (py + h / 2.0) / img_h
    nw = w / float(img_w)
    nh = h / float(img_h)
    return f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


@dataclass
class GenerationStats:
    saved: int = 0
    skipped_load: int = 0
    skipped_place: int = 0
    errors: list[str] = field(default_factory=list)


def label_stem_key(stem: str, delimiter: str) -> str:
    """
    Class name from a file stem: substring before the first ``delimiter``.

    If ``delimiter`` is empty, the full ``stem`` is the class (one file = one class name).
    If ``delimiter`` is not found, the full ``stem`` is used. If the prefix would be empty,
    the full ``stem`` is used.
    """
    if not stem:
        return stem
    if not delimiter:
        return stem
    i = stem.find(delimiter)
    if i < 0:
        return stem
    key = stem[:i]
    return key if key else stem


def build_class_mapping(
    crop_paths: list[Path], label_stem_delimiter: str
) -> tuple[dict[str, int], list[str]]:
    """Build a class mapping from the crop paths (optionally grouping by ``label_stem_key``)."""
    names = sorted({label_stem_key(p.stem, label_stem_delimiter) for p in crop_paths})
    name_to_id = {name: i for i, name in enumerate(names)}
    return name_to_id, names


def _stems_in_folder(folder: Path, suffix: str) -> set[str]:
    """Get the stems in a folder with the given suffix."""
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.glob(f"*{suffix}") if p.is_file()}


def _collect_used_stems(images_dir: Path, labels_dir: Path | None) -> set[str]:
    """Collect the used stems from the images and labels directories."""
    s = _stems_in_folder(images_dir, ".png")
    if labels_dir is not None:
        s |= _stems_in_folder(labels_dir, ".txt")
    return s


def unique_random_stem(used: set[str], rng: random.Random) -> str:
    """Generate a unique random stem."""
    for _ in range(100_000):
        n = rng.randint(0, 9_999_999_999)
        stem = f"{n:010d}"
        if stem not in used:
            used.add(stem)
            return stem
    raise RuntimeError("Could not allocate a unique 10-digit file name")


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> None:
    """Write a Ultralytics-style ``dataset.yaml`` at the dataset root (replaces ``classes.txt``)."""
    path_str = output_dir.resolve().as_posix()
    lines = [
        f"path: {json.dumps(path_str)}",
        "train: train/images",
        "val: val/images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {json.dumps(name)}")
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset_generation(
    crops_dir: Path,
    backgrounds_dir: Path,
    output_dir: Path,
    scale_pct: int,
    train_size: int,
    val_size: int,
    label_format: str,
    output_width: int = 640,
    output_height: int = 640,
    label_stem_delimiter: str = ".",
    crops_per_image: int = 1,
    max_overlap_percent: int = 100,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
    seed: int | None = None,
) -> GenerationStats:
    """Run the dataset generation."""
    stats = GenerationStats()
    rng = random.Random(seed)

    ow = max(1, int(output_width))
    oh = max(1, int(output_height))

    crop_paths = list_images_in_dir(crops_dir)
    bg_paths = list_images_in_tree(backgrounds_dir)
    if not crop_paths or not bg_paths:
        stats.errors.append("Need at least one crop image and one background image.")
        return stats

    if label_format not in (FORMAT_YOLO, FORMAT_YOLO_OBB):
        stats.errors.append("Label format must be YOLO bounding box or YOLO oriented bounding box.")
        return stats

    name_to_id, class_names = build_class_mapping(crop_paths, label_stem_delimiter)

    k = max(1, min(20, int(crops_per_image)))
    max_iou = max(0, min(100, int(max_overlap_percent))) / 100.0

    splits: list[tuple[str, int]] = [
        ("train", max(0, int(train_size))),
        ("val", max(0, int(val_size))),
    ]
    total_steps = sum(c for _, c in splits)
    if total_steps == 0:
        stats.errors.append("Training set size and test set size cannot both be zero.")
        return stats

    global_step = 0
    dataset_on_disk_started = False

    for split_name, split_count in splits:
        if split_count == 0:
            continue

        images_dir = output_dir / split_name / "images"
        labels_dir: Path | None = None
        if label_format in (FORMAT_YOLO, FORMAT_YOLO_OBB):
            labels_dir = output_dir / split_name / "labels"

        used_stems = _collect_used_stems(images_dir, labels_dir)

        schedule = balanced_crop_indices(split_count, len(crop_paths), rng)

        for step_idx, crop_idx in enumerate(schedule):
            if progress:
                progress(global_step, total_steps)
            global_step += 1
            if cancel_check and cancel_check():
                return stats

            if k == 1:
                index_list = [crop_idx]
            else:
                index_list = rng.choices(range(len(crop_paths)), k=k)

            cids: list[int] = []
            loaded: list[np.ndarray] = []
            for idx in index_list:
                pth = crop_paths[idx]
                im = load_image_bgra(pth)
                if im is None:
                    loaded = []
                    cids = []
                    break
                loaded.append(im)
                cn = label_stem_key(pth.stem, label_stem_delimiter)
                cids.append(name_to_id[cn])
            if not loaded or len(loaded) != k:
                stats.skipped_load += 1
                continue

            bg_path = rng.choice(bg_paths)
            bg = load_image_bgr(bg_path)
            if bg is None:
                stats.skipped_load += 1
                continue

            bh0, bw0 = bg.shape[:2]
            if bw0 < 1 or bh0 < 1:
                stats.skipped_load += 1
                continue
            interp = cv2.INTER_AREA if (ow * oh) < (bw0 * bh0) else cv2.INTER_LINEAR
            bg = cv2.resize(bg, (ow, oh), interpolation=interp)
            bh, bw = oh, ow

            box_sizes: list[tuple[int, int]] = []
            for crop in loaded:
                ch, cw = crop.shape[:2]
                r = scaled_crop_size(ch, cw, bh, bw, scale_pct)
                if r is None:
                    box_sizes = []
                    break
                box_sizes.append(r)
            if len(box_sizes) != k:
                stats.skipped_place += 1
                continue

            pos_list = try_place_k_rects_iou(box_sizes, bw, bh, max_iou, rng)
            if pos_list is None:
                stats.skipped_place += 1
                continue

            canvas = bg.copy()
            label_lines: list[str] = []
            ih, iw = oh, ow
            for ci in range(k):
                crop = loaded[ci]
                cid = cids[ci]
                new_w, new_h = box_sizes[ci]
                px, py = pos_list[ci]
                patch = resize_bgra(crop, new_w, new_h)
                alpha_blend_roi(canvas, patch, px, py)
                xmin_f = float(px)
                ymin_f = float(py)
                w_f = float(new_w)
                h_f = float(new_h)
                if label_format == FORMAT_YOLO:
                    label_lines.append(yolo_bbox_line(cid, xmin_f, ymin_f, w_f, h_f, iw, ih))
                else:
                    corners = rect_corners_xyxy(xmin_f, ymin_f, w_f, h_f)
                    label_lines.append(yolo_obb_line(cid, corners, iw, ih))

            stem_num = unique_random_stem(used_stems, rng)
            img_filename = f"{stem_num}.png"
            out_img_path = images_dir / img_filename

            try:
                ok, buf = cv2.imencode(".png", canvas)
                if not ok:
                    raise OSError("encode failed")
                if not dataset_on_disk_started:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    write_dataset_yaml(output_dir, class_names)
                    dataset_on_disk_started = True
                images_dir.mkdir(parents=True, exist_ok=True)
                if labels_dir is not None:
                    labels_dir.mkdir(parents=True, exist_ok=True)
                buf.tofile(str(out_img_path))
            except OSError as e:
                used_stems.discard(stem_num)
                stats.errors.append(f"{split_name}/{img_filename}: {e}")
                continue

            if labels_dir is not None:
                (labels_dir / f"{stem_num}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            stats.saved += 1

    return stats
