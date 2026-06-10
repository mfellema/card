"""
CARD — 2x2 grid image viewer with RGB channel toggles and largest-contour overlay.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from pathlib import Path

import cv2
from dataset_create import (
    list_images_in_tree,
    rect_corners_xyxy,
    run_dataset_generation,
    yolo_bbox_line,
    yolo_obb_line,
)
import numpy as np
from PySide6.QtCore import QObject, QSettings, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


# Save format labels (persisted in QSettings)
FORMAT_YOLO = "YOLO bounding box"
FORMAT_YOLO_OBB = "YOLO oriented bounding box"
SAVE_FORMATS = (FORMAT_YOLO, FORMAT_YOLO_OBB)

PROJECT_ROOT = Path(__file__).resolve().parent

APP_VERSION = "0.2.2"
APP_COPYRIGHT = "Copyright © Michael Fellema"

SETTINGS_ORG = "CARD"
SETTINGS_APP = "CARD"
KEY_SAVE_FORMAT = "saveFormat"
KEY_OUTPUT_DIRECTORY = "outputDirectory"
KEY_TRAINING_SET_SIZE = "trainingSetSize"
KEY_TEST_SET_SIZE = "testSetSize"
KEY_BACKGROUNDS_DIRECTORY = "backgroundsDirectory"
KEY_LABEL_STEM_DELIMITER = "labelStemDelimiter"
KEY_VIEW_SLOTS = ("view1", "view2", "view3", "view4")

# View slot values persisted in QSettings (stable keys)
VIEW_ORIGINAL = "original"
VIEW_ADJUSTED = "adjusted"
VIEW_CONTOUR_MASK = "contour_mask"
VIEW_CONTOUR_LINES = "contour_lines"
VIEW_BBOX_OVERLAY = "bbox_overlay"
VIEW_BBOX_CONTENTS = "bbox_contents"
VIEW_MODE_KEYS: tuple[str, ...] = (
    VIEW_ORIGINAL,
    VIEW_ADJUSTED,
    VIEW_CONTOUR_MASK,
    VIEW_CONTOUR_LINES,
    VIEW_BBOX_OVERLAY,
    VIEW_BBOX_CONTENTS,
)
VIEW_MODE_DISPLAY: dict[str, str] = {
    VIEW_ORIGINAL: "Original image",
    VIEW_ADJUSTED: "Color adjusted image",
    VIEW_CONTOUR_MASK: "Contour mask",
    VIEW_CONTOUR_LINES: "Contour lines",
    VIEW_BBOX_OVERLAY: "Bounding box overlay",
    VIEW_BBOX_CONTENTS: "Bounding box contents",
}
DEFAULT_VIEW_SLOTS: tuple[str, str, str, str] = (
    VIEW_ORIGINAL,
    VIEW_ADJUSTED,
    VIEW_BBOX_OVERLAY,
    VIEW_CONTOUR_LINES,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
OVERLAY_LINE_THICKNESS = 5

LIST_LOAD_DEBOUNCE_MS = 100
IMAGE_LRU_MAX = 6


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert BGR to RGB for display."""
    if bgr is None or bgr.size == 0:
        return QImage()
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    bytes_per_line = 3 * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def bgra_to_qimage(bgra: np.ndarray) -> QImage:
    """Convert BGRA to RGBA for display."""
    if bgra is None or bgra.size == 0:
        return QImage()
    rgba = np.ascontiguousarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
    h, w = rgba.shape[:2]
    bytes_per_line = 4 * w
    return QImage(rgba.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()


def apply_channel_mask(bgr: np.ndarray, r_on: bool, g_on: bool, b_on: bool) -> np.ndarray:
    """Disable color channels when toggled off (checkboxes: B, G, R order in array is B,G,R).
    This can be used to isolate the object from the background."""
    out = bgr.copy()
    if not b_on:
        out[:, :, 0] = 0
    if not g_on:
        out[:, :, 1] = 0
    if not r_on:
        out[:, :, 2] = 0
    return out


def bbox_axis_aligned_valid(img_w: int, img_h: int, x: int, y: int, w: int, h: int) -> bool:
    """Test to see if the axis-aligned bounding box is touching the edge of the image."""
    return x > 0 and y > 0 and (x + w) < img_w and (y + h) < img_h


def largest_contour_mask(gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Otsu threshold, find largest contour; return (binary_mask_uint8, contour_or_None)."""
    if gray is None or gray.size == 0:
        return None, None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1:
        return None, None
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask, largest


def axis_aligned_bbox_from_contour(contour: np.ndarray) -> tuple[int, int, int, int]:
    """Get the axis-aligned bounding box that tightly encloses a contour."""
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(w), int(h)


def oriented_box_points(contour: np.ndarray) -> np.ndarray:
    """Get the points of the oriented bounding box that tightly encloses a contour."""
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


def order_box_points(pts: np.ndarray) -> np.ndarray:
    """Order quad corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_obb_warp_matrix_and_size(box_pts: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    """Perspective from OBB quad to upright (w,h); returns (M, width, height) or None if degenerate."""
    pts = order_box_points(box_pts)
    tl, tr, br, bl = pts
    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    max_w = max(int(round(width_a)), int(round(width_b)))
    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    max_h = max(int(round(height_a)), int(round(height_b)))
    if max_w < 1 or max_h < 1:
        return None
    dst = np.array(
        [
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(pts, dst)
    return m, max_w, max_h


def crop_oriented_box_patch(bgr: np.ndarray, box_pts: np.ndarray) -> np.ndarray | None:
    """Perspective-warp the interior of the OBB quad to an upright rectangle; None if degenerate."""
    r = get_obb_warp_matrix_and_size(box_pts)
    if r is None:
        return None
    m, max_w, max_h = r
    return cv2.warpPerspective(
        bgr, m, (max_w, max_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def crop_obb_patch_masked_by_contour(
    bgr: np.ndarray, box_pts: np.ndarray, contour_filled_mask: np.ndarray | None
) -> np.ndarray | None:
    """
    OBB perspective crop; warped filled-contour region is opaque, rest of the warped quad is
    transparent (BGRA). If ``contour_filled_mask`` is None, returns BGR only (no alpha).
    """
    r = get_obb_warp_matrix_and_size(box_pts)
    if r is None:
        return None
    m, max_w, max_h = r
    patch = cv2.warpPerspective(
        bgr, m, (max_w, max_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    if contour_filled_mask is None:
        return patch
    mask_w = cv2.warpPerspective(
        contour_filled_mask,
        m,
        (max_w, max_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    _, mask_bin = cv2.threshold(mask_w, 127, 255, cv2.THRESH_BINARY)
    b_chan, g_chan, r_chan = cv2.split(patch)
    return cv2.merge([b_chan, g_chan, r_chan, mask_bin])


def write_bgr_image(path: str | Path, img: np.ndarray) -> None:
    """Write BGR or BGRA (PNG alpha). JPEG/WebP ignore extra alpha if used."""
    p = Path(path)
    ext = (p.suffix.lower() if p.suffix else ".png") or ".png"
    if ext in (".jpeg", ".jpe"):
        ext = ".jpg"
    if img.ndim == 3 and img.shape[2] == 4:
        if ext not in (".png", ".webp", ".tif", ".tiff"):
            ext = ".png"
            p = p.with_suffix(".png")
    elif ext not in (".png", ".jpg", ".bmp", ".tif", ".tiff", ".webp"):
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise OSError("Could not encode image")
    buf.tofile(str(p))


def read_save_format() -> str:
    """Get the save format from the settings."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    v = s.value(KEY_SAVE_FORMAT, FORMAT_YOLO_OBB)
    if isinstance(v, str) and v in SAVE_FORMATS:
        return v
    return FORMAT_YOLO_OBB


def write_save_format(fmt: str) -> None:
    """Set the save format in the settings."""
    if fmt not in SAVE_FORMATS:
        return
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    s.setValue(KEY_SAVE_FORMAT, fmt)


def read_output_directory() -> str:
    """Get the output directory from the settings."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    v = s.value(KEY_OUTPUT_DIRECTORY, "")
    return v if isinstance(v, str) else ""


def write_output_directory(path: str) -> None:
    """Set the output directory in the settings."""
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(KEY_OUTPUT_DIRECTORY, path)


def read_training_set_size() -> int:
    """Get the training set size from the settings."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    v = s.value(KEY_TRAINING_SET_SIZE, 750)
    try:
        n = int(v) if v is not None else 750
    except (TypeError, ValueError):
        n = 750
    return max(0, min(n, 9_999_999))


def write_training_set_size(n: int) -> None:
    """Set the training set size in the settings."""
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(
        KEY_TRAINING_SET_SIZE, max(0, min(int(n), 9_999_999))
    )


def read_test_set_size() -> int:
    """Get the test set size from the settings."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    v = s.value(KEY_TEST_SET_SIZE, 250)
    try:
        n = int(v) if v is not None else 250
    except (TypeError, ValueError):
        n = 250
    return max(0, min(n, 9_999_999))


def write_test_set_size(n: int) -> None:
    """Set the test set size in the settings."""
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(KEY_TEST_SET_SIZE, max(0, min(int(n), 9_999_999)))


def read_label_stem_delimiter() -> str:
    """Delimiter for class names: stem is split on its first match (empty => use full stem)."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    v = s.value(KEY_LABEL_STEM_DELIMITER, ".")
    return v if isinstance(v, str) else "."


def write_label_stem_delimiter(d: str) -> None:
    """Persist the class label stem delimiter (see ``read_label_stem_delimiter``)."""
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(KEY_LABEL_STEM_DELIMITER, d)


def default_dataset_output_directory() -> str:
    """Default synthetic dataset root: ``<save_root>/dataset``."""
    root = read_output_directory().strip()
    if not root:
        return ""
    return str(Path(root) / "dataset")


def normalize_view_mode_key(key: str | None) -> str:
    """What view mode is the user currently using?"""
    if isinstance(key, str) and key in VIEW_MODE_KEYS:
        return key
    return VIEW_ORIGINAL


def read_view_slots() -> list[str]:
    """Get the view slots from the settings."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    out: list[str] = []
    for i, k in enumerate(KEY_VIEW_SLOTS):
        v = s.value(k, DEFAULT_VIEW_SLOTS[i])
        out.append(normalize_view_mode_key(v if isinstance(v, str) else None))
    return out


def write_view_slots(slots: list[str]) -> None:
    """Set the view slots in the settings."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    for i, k in enumerate(KEY_VIEW_SLOTS):
        key = normalize_view_mode_key(slots[i] if i < len(slots) else None)
        s.setValue(k, key)


def compute_export_crop(
    bgr: np.ndarray,
    r_on: bool,
    g_on: bool,
    b_on: bool,
    fmt: str,
) -> tuple[np.ndarray | None, str | None]:
    """Shared crop geometry for PNG export and label files. Errors: no_contour, bad_patch, edge_bbox."""
    if fmt == FORMAT_YOLO:
        return axis_aligned_crop_from_bgr(bgr, r_on, g_on, b_on)
    return obb_crop_patch_from_bgr(bgr, r_on, g_on, b_on)


def export_crop_to_disk(
    bgr: np.ndarray,
    stem: str,
    crops_dir: Path,
    r_on: bool,
    g_on: bool,
    b_on: bool,
    fmt: str,
) -> tuple[bool, str | None]:
    """Write ``crops_dir / f"{stem}.png"`` only (no label file)."""
    crop, err = compute_export_crop(bgr, r_on, g_on, b_on, fmt)
    if err is not None:
        return False, err
    assert crop is not None
    out_img = crops_dir / f"{stem}.png"
    write_bgr_image(out_img, crop)
    return True, None


def write_yolo_label_for_image(
    bgr: np.ndarray,
    stem: str,
    crops_dir: Path,
    r_on: bool,
    g_on: bool,
    b_on: bool,
    fmt: str,
) -> tuple[bool, str | None]:
    """Write ``crops_dir / f"{stem}.txt"`` from the same crop geometry as export (full-frame class 0)."""
    crop, err = compute_export_crop(bgr, r_on, g_on, b_on, fmt)
    if err is not None:
        return False, err
    assert crop is not None
    h, w = crop.shape[:2]
    if fmt == FORMAT_YOLO:
        line = yolo_bbox_line(0, 0.0, 0.0, float(w), float(h), w, h)
    else:
        corners = rect_corners_xyxy(0.0, 0.0, float(w), float(h))
        line = yolo_obb_line(0, corners, w, h)
    (crops_dir / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")
    return True, None


def default_backgrounds_directory() -> str:
    """Default backgrounds directory: ``<project_root>/backgrounds``."""
    return str(PROJECT_ROOT / "backgrounds")


def read_backgrounds_directory() -> str:
    """Get the backgrounds directory from the settings."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    v = s.value(KEY_BACKGROUNDS_DIRECTORY, "")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return default_backgrounds_directory()


def write_backgrounds_directory(path: str) -> None:
    """Set the backgrounds directory in the settings."""
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(KEY_BACKGROUNDS_DIRECTORY, path)


def list_images_in_dir(directory: str | Path) -> list[Path]:
    """List all images in a directory."""
    p = Path(directory)
    if not p.is_dir():
        return []
    out: list[Path] = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(f)
    return out


def load_image_bgr(path: str) -> np.ndarray | None:
    """Decode BGR image from disk; same as synchronous load in ``MainWindow``."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path)
    if img is None or img.size == 0:
        return None
    return img


class ImageLoadWorker(QObject):
    """Decodes images on a background thread. MainWindow owns the QThread."""

    fullReady = Signal(str, int, object)
    loadFailed = Signal(str, int, str)
    prefetchReady = Signal(str, object)  # path, bgr

    @Slot(str, int)
    def loadForDisplay(self, path: str, display_token: int) -> None:
        bgr = load_image_bgr(path)
        if bgr is None or bgr.size == 0:
            self.loadFailed.emit(path, display_token, "Could not load the selected file.")
            return
        self.fullReady.emit(path, display_token, bgr)

    @Slot(str)
    def loadPrefetch(self, path: str) -> None:
        bgr = load_image_bgr(path)
        if bgr is None or bgr.size == 0:
            return
        self.prefetchReady.emit(path, bgr)


def obb_crop_patch_from_bgr(
    bgr: np.ndarray, r_on: bool, g_on: bool, b_on: bool
) -> tuple[np.ndarray | None, str | None]:
    """
    Channel mask → largest contour → min-area OBB → masked warp.
    Contour/geometry use the color-adjusted image; the saved patch pixels come from the original
    ``bgr``. On failure returns (None, 'no_contour' | 'bad_patch' | 'edge_bbox').
    """
    adjusted = apply_channel_mask(bgr, r_on, g_on, b_on)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    contour_mask, cnt = largest_contour_mask(gray)
    if cnt is None:
        return None, "no_contour"
    ih, iw = adjusted.shape[:2]
    x, y, w, h = axis_aligned_bbox_from_contour(cnt)
    if not bbox_axis_aligned_valid(iw, ih, x, y, w, h):
        return None, "edge_bbox"
    obb_pts = oriented_box_points(cnt)
    patch = crop_obb_patch_masked_by_contour(bgr, obb_pts, contour_mask)
    if patch is None or patch.size == 0:
        return None, "bad_patch"
    return patch, None


def axis_aligned_crop_from_bgr(
    bgr: np.ndarray, r_on: bool, g_on: bool, b_on: bool
) -> tuple[np.ndarray | None, str | None]:
    """Axis-aligned bounding-rect crop; rejects boxes touching the image edge. Pixels are from
    the original ``bgr``; channel toggles only affect detection."""
    adjusted = apply_channel_mask(bgr, r_on, g_on, b_on)
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    _mask, cnt = largest_contour_mask(gray)
    if cnt is None:
        return None, "no_contour"
    ih, iw = adjusted.shape[:2]
    x, y, w, h = axis_aligned_bbox_from_contour(cnt)
    if w < 1 or h < 1:
        return None, "bad_patch"
    if not bbox_axis_aligned_valid(iw, ih, x, y, w, h):
        return None, "edge_bbox"
    crop = bgr[y : y + h, x : x + w].copy()
    return crop, None


class SettingsDialog(QDialog):
    """Settings dialog for the application."""
    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the settings dialog."""
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(560, 420)

        self._categories = QListWidget()
        self._categories.addItem("Save")
        self._categories.addItem("Views")
        self._stack = QStackedWidget()

        save_page = QWidget()
        save_form = QFormLayout(save_page)
        self._output_edit = QLineEdit(read_output_directory())
        self._output_edit.setPlaceholderText(
            "Save root; crops/ and dataset/ are created when you save crops or generate a dataset"
        )
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output)
        out_row = QWidget()
        out_lay = QHBoxLayout(out_row)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.addWidget(self._output_edit, stretch=1)
        out_lay.addWidget(browse_btn)
        save_form.addRow("Output directory:", out_row)

        self._save_format_combo = QComboBox()
        for f in SAVE_FORMATS:
            self._save_format_combo.addItem(f)
        current = read_save_format()
        idx = self._save_format_combo.findText(current)
        if idx >= 0:
            self._save_format_combo.setCurrentIndex(idx)
        save_form.addRow("Save format:", self._save_format_combo)

        self._label_stem_delim_edit = QLineEdit(read_label_stem_delimiter())
        self._label_stem_delim_edit.setToolTip(
            "Synthetic dataset: class name = crop filename (without extension) up to the first "
            "match of this delimiter. Empty = use the full filename stem. Default is a period, "
            "so e.g. card.001.png and card.002.png share the class name card."
        )
        save_form.addRow("Class label stem delimiter:", self._label_stem_delim_edit)
        self._stack.addWidget(save_page)

        views_page = QWidget()
        views_form = QFormLayout(views_page)
        self._view_combos: list[QComboBox] = []
        slots = read_view_slots()
        for i in range(4):
            cb = QComboBox()
            for key in VIEW_MODE_KEYS:
                cb.addItem(VIEW_MODE_DISPLAY[key], key)
            want = slots[i] if i < len(slots) else DEFAULT_VIEW_SLOTS[i]
            ix = cb.findData(want, Qt.ItemDataRole.UserRole)
            if ix < 0:
                ix = 0
            cb.setCurrentIndex(ix)
            self._view_combos.append(cb)
            views_form.addRow(f"View {i + 1}:", cb)
        self._stack.addWidget(views_page)

        body = QHBoxLayout()
        body.addWidget(self._categories)
        body.addWidget(self._stack, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(body)
        main_layout.addWidget(buttons)

        self._categories.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._categories.setCurrentRow(0)

    def _browse_output(self) -> None:
        """Browse for the output directory."""
        d = QFileDialog.getExistingDirectory(self, "Output directory", self._output_edit.text() or "")
        if d:
            self._output_edit.setText(d)

    def accept(self) -> None:
        """Save the settings."""
        write_save_format(self._save_format_combo.currentText())
        write_label_stem_delimiter(self._label_stem_delim_edit.text())
        out = self._output_edit.text().strip()
        write_output_directory(out)
        slot_values: list[str] = []
        for cb in self._view_combos:
            d = cb.currentData(Qt.ItemDataRole.UserRole)
            slot_values.append(normalize_view_mode_key(str(d) if d is not None else None))
        write_view_slots(slot_values)
        super().accept()


class CreateDatasetDialog(QDialog):
    """Confirm paths, scale, and dataset size before synthetic dataset generation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create dataset")
        self.resize(560, 420)

        out_root = read_output_directory().strip()
        default_crops = str(Path(out_root) / "crops") if out_root else ""
        default_out = default_dataset_output_directory()

        self._crops_edit = QLineEdit(default_crops)
        self._crops_edit.setPlaceholderText("Folder containing exported crop PNGs (e.g. …/crops)")
        crops_btn = QPushButton("Browse…")
        crops_btn.clicked.connect(self._browse_crops)
        crops_row = QWidget()
        crops_lay = QHBoxLayout(crops_row)
        crops_lay.setContentsMargins(0, 0, 0, 0)
        crops_lay.addWidget(self._crops_edit, stretch=1)
        crops_lay.addWidget(crops_btn)

        self._bg_edit = QLineEdit(read_backgrounds_directory())
        self._bg_edit.setPlaceholderText("Background images folder (includes subfolders)")
        bg_btn = QPushButton("Browse…")
        bg_btn.clicked.connect(self._browse_bg)
        bg_row = QWidget()
        bg_lay = QHBoxLayout(bg_row)
        bg_lay.setContentsMargins(0, 0, 0, 0)
        bg_lay.addWidget(self._bg_edit, stretch=1)
        bg_lay.addWidget(bg_btn)

        self._out_edit = QLineEdit(default_out)
        self._out_edit.setPlaceholderText("Dataset root (e.g. …/output/dataset; train/ and val/ inside)")
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._browse_out)
        out_row = QWidget()
        out_lay = QHBoxLayout(out_row)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.addWidget(self._out_edit, stretch=1)
        out_lay.addWidget(out_btn)

        self._res_w = QSpinBox()
        self._res_w.setRange(32, 8192)
        self._res_w.setValue(640)
        self._res_h = QSpinBox()
        self._res_h.setRange(32, 8192)
        self._res_h.setValue(640)
        self._res_w.setToolTip("Width of each saved image after compositing (uniform resize).")
        self._res_h.setToolTip("Height of each saved image after compositing (uniform resize).")
        res_row = QWidget()
        res_lay = QHBoxLayout(res_row)
        res_lay.setContentsMargins(0, 0, 0, 0)
        res_lay.addWidget(self._res_w)
        res_lay.addWidget(QLabel("×"))
        res_lay.addWidget(self._res_h)

        self._scale_spin = QSpinBox()
        self._scale_spin.setRange(10, 90)
        self._scale_spin.setValue(50)
        self._scale_spin.setToolTip(
            "After the background is resized to the chosen resolution, the crop is scaled "
            "to fit inside a box of this percent of output width and height (aspect ratio kept; "
            "larger value ⇒ larger pasted crop)."
        )

        self._crops_per_image_spin = QSpinBox()
        self._crops_per_image_spin.setRange(1, 20)
        self._crops_per_image_spin.setValue(1)
        self._crops_per_image_spin.setToolTip("How many crop PNGs to paste onto each synthetic image (random placement).")
        self._max_overlap_spin = QSpinBox()
        self._max_overlap_spin.setRange(0, 100)
        self._max_overlap_spin.setValue(100)
        self._max_overlap_spin.setSuffix(" %")
        self._max_overlap_spin.setToolTip(
            "Maximum allowed IoU (intersection over union) between any two pasted crops, as a percent. "
            "100% means no limit; 0% means no overlapping area (touching is allowed)."
        )

        self._train_size_spin = QSpinBox()
        self._train_size_spin.setRange(0, 9_999_999)
        self._train_size_spin.setValue(read_training_set_size())
        self._test_size_spin = QSpinBox()
        self._test_size_spin.setRange(0, 9_999_999)
        self._test_size_spin.setValue(read_test_set_size())

        self._label_format_combo = QComboBox()
        self._label_format_combo.addItem(FORMAT_YOLO)
        self._label_format_combo.addItem(FORMAT_YOLO_OBB)
        self._label_format_combo.setCurrentIndex(0)

        self._label_stem_delim_edit = QLineEdit(read_label_stem_delimiter())
        self._label_stem_delim_edit.setToolTip(
            "Class name = each crop filename (without extension) up to the first match of this "
            "delimiter. Empty = use the full stem. Default is a period, so e.g. card.001 and "
            "card.002 share the class name card."
        )

        self._open_when_done = QCheckBox("Open output folder when finished")
        self._open_when_done.setChecked(True)

        form = QFormLayout()
        form.addRow("Crops directory:", crops_row)
        form.addRow("Backgrounds directory:", bg_row)
        form.addRow("Output directory:", out_row)
        form.addRow("Label format:", self._label_format_combo)
        form.addRow("Class label stem delimiter:", self._label_stem_delim_edit)
        form.addRow("Resolution (W × H):", res_row)
        form.addRow("Scale (% of background):", self._scale_spin)
        form.addRow("Crops per image:", self._crops_per_image_spin)
        form.addRow("Max overlap (IoU):", self._max_overlap_spin)
        form.addRow("Training set size:", self._train_size_spin)
        form.addRow("Test set size:", self._test_size_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(form)
        main_layout.addWidget(self._open_when_done)
        main_layout.addWidget(buttons)

    def _browse_crops(self) -> None:
        """Browse for the crops directory."""
        d = QFileDialog.getExistingDirectory(self, "Crops directory", self._crops_edit.text() or "")
        if d:
            self._crops_edit.setText(d)

    def _browse_bg(self) -> None:
        """Browse for the backgrounds directory."""
        d = QFileDialog.getExistingDirectory(self, "Backgrounds directory", self._bg_edit.text() or "")
        if d:
            self._bg_edit.setText(d)

    def _browse_out(self) -> None:  
        """Browse for the output directory."""
        d = QFileDialog.getExistingDirectory(self, "Output directory", self._out_edit.text() or "")
        if d:
            self._out_edit.setText(d)

    def get_config(self) -> dict:
        """Get the configuration for the dataset creation."""
        return {
            "crops_dir": Path(self._crops_edit.text().strip()),
            "backgrounds_dir": Path(self._bg_edit.text().strip()),
            "output_dir": Path(self._out_edit.text().strip()),
            "label_format": self._label_format_combo.currentText(),
            "output_width": self._res_w.value(),
            "output_height": self._res_h.value(),
            "scale": self._scale_spin.value(),
            "train_size": self._train_size_spin.value(),
            "test_size": self._test_size_spin.value(),
            "open_when_done": self._open_when_done.isChecked(),
            "label_stem_delimiter": self._label_stem_delim_edit.text(),
            "crops_per_image": self._crops_per_image_spin.value(),
            "max_overlap_percent": self._max_overlap_spin.value(),
        }

    def accept(self) -> None:
        """Create the dataset."""
        crops = self._crops_edit.text().strip()
        bg = self._bg_edit.text().strip()
        out = self._out_edit.text().strip()
        if not crops or not bg or not out:
            QMessageBox.warning(self, "Create dataset", "Set crops, backgrounds, and output directories.")
            return
        cp, bp = Path(crops), Path(bg)
        if not cp.is_dir():
            QMessageBox.warning(self, "Create dataset", "Crops directory does not exist or is not a folder.")
            return
        if not bp.is_dir():
            QMessageBox.warning(
                self, "Create dataset", "Backgrounds directory does not exist or is not a folder."
            )
            return
        if not list_images_in_dir(cp):
            QMessageBox.warning(self, "Create dataset", "No supported images found in the crops folder.")
            return
        if not list_images_in_tree(bp):
            QMessageBox.warning(
                self,
                "Create dataset",
                "No supported images found in the backgrounds folder (including subfolders).",
            )
            return
        if self._train_size_spin.value() + self._test_size_spin.value() < 1:
            QMessageBox.warning(
                self, "Create dataset", "Training set size and test set size cannot both be zero."
            )
            return
        write_training_set_size(self._train_size_spin.value())
        write_test_set_size(self._test_size_spin.value())
        write_backgrounds_directory(self._bg_edit.text().strip())
        write_label_stem_delimiter(self._label_stem_delim_edit.text())
        super().accept()


class MainWindow(QMainWindow):
    """Main window for the application."""

    _request_image_display = Signal(str, int)
    _request_image_prefetch = Signal(str)

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        try:
            Path(default_backgrounds_directory()).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.setWindowTitle("CARD")
        self._image_bgr: np.ndarray | None = None
        self._source_path: str | None = None
        self._image_lru: OrderedDict[str, np.ndarray] = OrderedDict()
        self._display_token: int = 0
        self._last_largest_contour: np.ndarray | None = None
        self._save_format: str = read_save_format()
        self._bbox_xywh: tuple[int, int, int, int] | None = None
        self._obb_points_px: np.ndarray | None = None
        self._image_paths: list[Path] = []
        self._export_w = 0
        self._export_h = 0
        self._view_slots: list[str] = read_view_slots()

        self._list_debounce = QTimer(self)
        self._list_debounce.setSingleShot(True)
        self._list_debounce.setInterval(LIST_LOAD_DEBOUNCE_MS)
        self._list_debounce.timeout.connect(self._on_list_debounce_timeout)
        self._pending_debounce_path: str | None = None

        self._load_thread = QThread(self)
        self._load_worker = ImageLoadWorker()
        self._load_worker.moveToThread(self._load_thread)
        self._request_image_display.connect(
            self._load_worker.loadForDisplay, Qt.ConnectionType.QueuedConnection
        )
        self._request_image_prefetch.connect(
            self._load_worker.loadPrefetch, Qt.ConnectionType.QueuedConnection
        )
        self._load_worker.fullReady.connect(self._on_image_full_ready, Qt.ConnectionType.QueuedConnection)
        self._load_worker.loadFailed.connect(self._on_image_load_failed, Qt.ConnectionType.QueuedConnection)
        self._load_worker.prefetchReady.connect(self._on_image_prefetch_ready, Qt.ConnectionType.QueuedConnection)
        self._load_thread.start()

        grid_wrap = QWidget()
        grid = QGridLayout(grid_wrap)
        grid.setSpacing(4)

        self._label_tl = self._make_cell_label()
        self._label_tr = self._make_cell_label()
        self._label_bl = self._make_cell_label()
        self._label_br = self._make_cell_label()

        grid.addWidget(self._label_tl, 0, 0)
        grid.addWidget(self._label_tr, 0, 1)
        grid.addWidget(self._label_bl, 1, 0)
        grid.addWidget(self._label_br, 1, 1)

        self._file_list = QListWidget()
        self._file_list.setMinimumWidth(220)
        self._file_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._file_list.currentRowChanged.connect(self._on_file_list_row_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(grid_wrap)
        splitter.addWidget(self._file_list)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self._ch_r = QCheckBox("R")
        self._ch_g = QCheckBox("G")
        self._ch_b = QCheckBox("B")
        for ch in (self._ch_r, self._ch_g, self._ch_b):
            ch.setChecked(True)
            ch.setToolTip("Channel on (checked) or off (unchecked)")
            ch.toggled.connect(self._on_channel_toggled)

        ribbon = QWidget()
        ribbon_layout = QHBoxLayout(ribbon)
        ribbon_layout.setContentsMargins(8, 4, 8, 4)
        ribbon_layout.addWidget(QLabel("RGB:"))
        for ch in (self._ch_r, self._ch_g, self._ch_b):
            ribbon_layout.addWidget(ch)
        rgb_toolbar = QToolBar("RGB")
        rgb_toolbar.addWidget(ribbon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, rgb_toolbar)

        self._build_menu()
        self._setup_shortcuts()

    def _setup_shortcuts(self) -> None:
        """Setup the shortcuts for the application."""
        sc_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        sc_space.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_space.activated.connect(self._on_space_shortcut)
        sc_up = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        sc_up.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_up.activated.connect(partial(self._step_file_list, -1))
        sc_dn = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        sc_dn.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_dn.activated.connect(partial(self._step_file_list, 1))

    def _on_space_shortcut(self) -> None:
        """Save the crop and move to the next image."""
        if self._save_annotation():
            self._go_to_next_image()

    def _step_file_list(self, delta: int) -> None:
        """Step through the file list."""
        if not self._image_paths:
            return
        row = self._file_list.currentRow()
        n = len(self._image_paths)
        new_row = max(0, min(n - 1, row + delta))
        if new_row == row:
            return
        self._file_list.setCurrentRow(new_row)

    def _on_file_list_row_changed(self, row: int) -> None:
        """Debounce file list selection; load runs in _on_list_debounce_timeout."""
        if row < 0 or row >= len(self._image_paths):
            return
        path = str(self._image_paths[row].resolve())
        cur = str(Path(self._source_path).resolve()) if self._source_path else ""
        if path == cur:
            return
        self._pending_debounce_path = path
        self._list_debounce.start()

    def _on_list_debounce_timeout(self) -> None:
        """Load the image at the current row."""
        if self._pending_debounce_path is None:
            return
        path = self._pending_debounce_path
        self._pending_debounce_path = None
        row = self._file_list.currentRow()
        if row < 0 or row >= len(self._image_paths):
            return
        if str(self._image_paths[row].resolve()) != path:
            return
        self._begin_async_list_load(str(self._image_paths[row]))

    @staticmethod
    def _lru_key(path: str) -> str:
        """Get the key for the image in the LRU cache."""
        return str(Path(path).resolve())

    def _lru_put(self, key: str, bgr: np.ndarray) -> None:
        """Put the image into the LRU cache."""
        self._image_lru[key] = bgr
        self._image_lru.move_to_end(key)
        while len(self._image_lru) > IMAGE_LRU_MAX:
            self._image_lru.popitem(last=False)

    def _begin_async_list_load(self, path: str) -> None:
        """Start loading path from list: LRU hit or request worker (full decode)."""
        self._display_token += 1
        tok = self._display_token
        key = self._lru_key(path)
        if key in self._image_lru:
            self._image_lru.move_to_end(key)
            self._image_bgr = self._image_lru[key]
            self._source_path = path
            self._update_views()
            self._prefetch_neighbors()
            return
        self._request_image_display.emit(path, tok)

    def _prefetch_neighbors(self) -> None:
        """Prefetch the neighbors of the current image."""
        if not self._image_paths or not self._source_path:
            return
        key_cur = self._lru_key(self._source_path)
        try:
            idx = next(
                i
                for i, p in enumerate(self._image_paths)
                if self._lru_key(str(p)) == key_cur
            )
        except StopIteration:
            return
        for delta in (-1, 1, -2, 2):
            j = idx + delta
            if 0 <= j < len(self._image_paths):
                p = str(self._image_paths[j].resolve())
                if self._lru_key(p) in self._image_lru:
                    continue
                self._request_image_prefetch.emit(p)

    def _on_image_full_ready(self, path: str, token: int, bgr: object) -> None:
        """Handle the full image decode."""
        if token != self._display_token:
            return
        if not isinstance(bgr, np.ndarray):
            return
        self._lru_put(self._lru_key(path), bgr)
        self._image_bgr = bgr
        self._source_path = path
        self._update_views()
        self._prefetch_neighbors()

    def _on_image_load_failed(self, path: str, token: int, message: str) -> None:
        """Handle the image load failure."""
        if token != self._display_token:
            return
        QMessageBox.warning(self, "Open image", message)

    def _on_image_prefetch_ready(self, path: str, bgr: object) -> None:
        """Handle the image prefetch."""
        if not isinstance(bgr, np.ndarray):
            return
        k = self._lru_key(path)
        if k in self._image_lru:
            return
        self._lru_put(k, bgr)

    def _refresh_file_list(self, directory: str | Path | None, select_path: str | None) -> None:
        """Refresh the file list."""
        self._file_list.blockSignals(True)
        self._file_list.clear()
        self._image_paths = list_images_in_dir(directory) if directory else []
        for p in self._image_paths:
            self._file_list.addItem(p.name)
        if select_path:
            sel = str(Path(select_path).resolve())
            for i, p in enumerate(self._image_paths):
                if str(p.resolve()) == sel:
                    self._file_list.setCurrentRow(i)
                    break
        self._file_list.blockSignals(False)

    def _load_image_path_immediate(self, path: str, refresh_list: bool = True) -> bool:
        """Load synchronously (Open directory / first image). Updates LRU, no debounce."""
        self._list_debounce.stop()
        self._pending_debounce_path = None
        img = load_image_bgr(path)
        if img is None or img.size == 0:
            QMessageBox.warning(self, "Open image", "Could not load the selected file.")
            return False
        self._display_token += 1
        self._lru_put(self._lru_key(path), img)
        self._image_bgr = img
        self._source_path = path
        if refresh_list:
            self._refresh_file_list(Path(path).parent, path)
        self._update_views()
        self._prefetch_neighbors()
        return True

    def _go_to_next_image(self) -> None:
        """Go to the next image in the file list."""
        if not self._source_path or not self._image_paths:
            return
        try:
            idx = next(i for i, p in enumerate(self._image_paths) if str(p) == self._source_path)
        except StopIteration:
            return
        if idx + 1 >= len(self._image_paths):
            return
        self._file_list.setCurrentRow(idx + 1)

    def _on_channel_toggled(self) -> None:
        """Update the views when the channel is toggled."""
        self._update_views()

    def _channel_mask(self) -> tuple[bool, bool, bool]:
        """Get the channel mask.""" 
        return self._ch_r.isChecked(), self._ch_g.isChecked(), self._ch_b.isChecked()

    def _open_settings(self) -> None:
        """Open the settings dialog."""
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._save_format = read_save_format()
            self._view_slots = read_view_slots()
            self._update_views()

    def _apply_source_output_layout(self, source_dir: Path) -> None:
        """Apply the source and output layout."""
        out = (source_dir / "output").resolve()
        write_output_directory(str(out))

    def _make_cell_label(self) -> QLabel:
        """Make a cell label for the image viewer."""
        lab = QLabel()
        lab.setMinimumSize(320, 240)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.setStyleSheet("background-color: #2a2a2a; color: #888;")
        lab.setText("No image")
        lab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lab.setScaledContents(False)
        return lab

    def _build_menu(self) -> None:
        """Build the menu for the application."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        act_open = QAction("Open &directory…", self)
        act_open.triggered.connect(self._open_directory)
        file_menu.addAction(act_open)

        act_open_images = QAction("Open &image(s)…", self)
        act_open_images.triggered.connect(self._open_images)
        file_menu.addAction(act_open_images)

        act_save = QAction("&Save crop…", self)
        act_save.triggered.connect(self._save_annotation)
        file_menu.addAction(act_save)

        act_save_all = QAction("Save all &crops…", self)
        act_save_all.triggered.connect(self._save_all_crops)
        file_menu.addAction(act_save_all)

        act_label = QAction("Save &label…", self)
        act_label.triggered.connect(self._save_label)
        file_menu.addAction(act_label)

        act_labels_all = QAction("Save all &labels…", self)
        act_labels_all.triggered.connect(self._save_all_labels)
        file_menu.addAction(act_labels_all)

        act_dataset = QAction("Create dataset…", self)
        act_dataset.triggered.connect(self._create_dataset)
        file_menu.addAction(act_dataset)

        file_menu.addSeparator()

        act_close = QAction("&Close", self)
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

        edit_menu = menu_bar.addMenu("&Edit")
        act_settings = QAction("&Settings…", self)
        act_settings.triggered.connect(self._open_settings)
        edit_menu.addAction(act_settings)

        help_menu = menu_bar.addMenu("&Help")
        act_readme = QAction("&Readme", self)
        act_readme.triggered.connect(self._open_readme)
        help_menu.addAction(act_readme)
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _open_readme(self) -> None:
        """Open the readme file."""
        readme_path = PROJECT_ROOT / "readme.md"
        if not readme_path.is_file():
            QMessageBox.warning(
                self,
                "Readme",
                f"Could not find readme.md next to the application ({readme_path}).",
            )
            return
        url = QUrl.fromLocalFile(str(readme_path.resolve()))
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "Readme", "Could not open readme.md with the default application.")

    def _show_about(self) -> None:
        """Show the about dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("About CARD")
        dlg.resize(440, 240)
        layout = QVBoxLayout(dlg)
        about_lbl = QLabel(
            f"<b>CARD</b> — Crop, Annotate, Record Dataset.<br><br>"
            f"<b>Version</b> {APP_VERSION}<br><br>"
            f"{APP_COPYRIGHT}<br><br>"
            "Licensed under the <b>MIT License</b>."
        )
        about_lbl.setWordWrap(True)
        about_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(about_lbl)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def _open_directory(self) -> None:
        """Open a directory."""
        start = ""
        if self._source_path:
            start = str(Path(self._source_path).parent)
        elif read_output_directory().strip():
            start = read_output_directory().strip()
        d = QFileDialog.getExistingDirectory(self, "Open directory", start or str(PROJECT_ROOT))
        if not d:
            return
        self._apply_source_output_layout(Path(d))
        paths = list_images_in_dir(Path(d))
        if not paths:
            QMessageBox.information(
                self,
                "Open directory",
                "No supported images found in this folder.",
            )
            return
        self._load_image_path_immediate(str(paths[0]), refresh_list=True)

    def _open_images(self) -> None:
        """Open images."""
        start = ""
        if self._source_path:
            start = str(Path(self._source_path).parent)
        elif read_output_directory().strip():
            start = read_output_directory().strip()
        exts = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        filters = f"Images ({exts});;All files (*.*)"
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open image(s)", start or str(PROJECT_ROOT), filters
        )
        if not paths:
            return
        self._file_list.blockSignals(True)
        existing = {str(p.resolve()) for p in self._image_paths}
        first_new_row: int | None = None
        for s in paths:
            rp = str(Path(s).resolve())
            if rp in existing:
                continue
            p = Path(s)
            self._image_paths.append(p)
            self._file_list.addItem(p.name)
            existing.add(rp)
            if first_new_row is None:
                first_new_row = len(self._image_paths) - 1
        self._file_list.blockSignals(False)
        if first_new_row is None:
            QMessageBox.information(
                self,
                "Open image(s)",
                "Those images are already in the list.",
            )
            return
        self._apply_source_output_layout(Path(paths[0]).parent)
        self._file_list.setCurrentRow(first_new_row)

    def _save_annotation(self) -> bool:
        """Save the crop annotation."""
        if self._image_bgr is None:
            QMessageBox.information(self, "Save crop", "No image loaded.")
            return False
        out_root = read_output_directory().strip()
        if not out_root:
            QMessageBox.warning(
                self,
                "Save crop",
                "Set an output directory in Settings (Save).",
            )
            return False

        r_on, g_on, b_on = self._channel_mask()
        fmt = read_save_format()
        root = Path(out_root)
        crops_dir = root / "crops"
        try:
            crops_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Save crop", str(e))
            return False

        stem = Path(self._source_path).stem
        try:
            ok, err = export_crop_to_disk(self._image_bgr, stem, crops_dir, r_on, g_on, b_on, fmt)
        except OSError as e:
            QMessageBox.critical(self, "Save crop", str(e))
            return False
        if not ok:
            if err == "no_contour":
                QMessageBox.warning(
                    self,
                    "Save crop",
                    "No contour found. Adjust channel toggles or choose a different image.",
                )
            elif err == "edge_bbox":
                QMessageBox.warning(
                    self,
                    "Save crop",
                    "Axis-aligned box touches the image edge; crop is not valid for export.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Save crop",
                    "Crop region is too small to export.",
                )
            return False
        return True

    def _save_label(self) -> bool:
        """Save the label."""
        if self._image_bgr is None:
            QMessageBox.information(self, "Save label", "No image loaded.")
            return False
        out_root = read_output_directory().strip()
        if not out_root:
            QMessageBox.warning(
                self,
                "Save label",
                "Set an output directory in Settings (Save).",
            )
            return False
        r_on, g_on, b_on = self._channel_mask()
        fmt = read_save_format()
        root = Path(out_root)
        crops_dir = root / "crops"
        try:
            crops_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Save label", str(e))
            return False
        stem = Path(self._source_path).stem
        try:
            ok, err = write_yolo_label_for_image(
                self._image_bgr, stem, crops_dir, r_on, g_on, b_on, fmt
            )
        except OSError as e:
            QMessageBox.critical(self, "Save label", str(e))
            return False
        if not ok:
            if err == "no_contour":
                QMessageBox.warning(
                    self,
                    "Save label",
                    "No contour found. Adjust channel toggles or choose a different image.",
                )
            elif err == "edge_bbox":
                QMessageBox.warning(
                    self,
                    "Save label",
                    "Axis-aligned box touches the image edge; label is not valid.",
                )
            else:
                QMessageBox.warning(self, "Save label", "Crop region is too small for a label.")
            return False
        return True

    def _save_all_labels(self) -> None:
        """Save all labels."""
        if not self._image_paths:
            QMessageBox.information(
                self,
                "Save all labels",
                "Add images to the list (open a directory or open image(s)) first.",
            )
            return
        out_root = read_output_directory().strip()
        if not out_root:
            QMessageBox.warning(
                self,
                "Save all labels",
                "Set an output directory in Settings (Save).",
            )
            return
        fmt = read_save_format()
        root = Path(out_root)
        crops_dir = root / "crops"
        try:
            crops_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Save all labels", str(e))
            return

        r_on, g_on, b_on = self._channel_mask()
        paths = list(self._image_paths)
        n = len(paths)
        prog = QProgressDialog("Saving labels…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)

        saved = 0
        load_fail: list[str] = []
        skip_no_contour = 0
        skip_bad_patch = 0
        skip_edge = 0
        write_fail: list[str] = []

        for i, p in enumerate(paths):
            prog.setValue(i)
            if prog.wasCanceled():
                break
            prog.setLabelText(f"Label {p.name}…")
            QApplication.processEvents()

            bgr = load_image_bgr(str(p))
            if bgr is None:
                load_fail.append(p.name)
                continue

            try:
                ok, err = write_yolo_label_for_image(bgr, p.stem, crops_dir, r_on, g_on, b_on, fmt)
            except OSError as e:
                write_fail.append(f"{p.name}: {e}")
                continue
            if ok:
                saved += 1
            elif err == "no_contour":
                skip_no_contour += 1
            elif err == "edge_bbox":
                skip_edge += 1
            else:
                skip_bad_patch += 1

        prog.setValue(n)

        lines: list[str] = [f"Saved: {saved} ({fmt})"]
        if skip_no_contour:
            lines.append(f"Skipped (no contour): {skip_no_contour}")
        if skip_edge:
            lines.append(f"Skipped (bbox touches edge): {skip_edge}")
        if skip_bad_patch:
            lines.append(f"Skipped (crop too small or degenerate): {skip_bad_patch}")
        if load_fail:
            shown = load_fail[:12]
            extra = f" (+{len(load_fail) - len(shown)} more)" if len(load_fail) > len(shown) else ""
            lines.append(f"Load failed ({len(load_fail)}): {', '.join(shown)}{extra}")
        if write_fail:
            lines.append("Write errors:")
            lines.extend(write_fail[:8])
            if len(write_fail) > 8:
                lines.append(f"(+{len(write_fail) - 8} more)")

        QMessageBox.information(self, "Save all labels", "\n".join(lines))

    def _save_all_crops(self) -> None:
        """Save all crops."""
        if not self._image_paths:
            QMessageBox.information(
                self,
                "Save all crops",
                "Add images to the list (open a directory or open image(s)) first.",
            )
            return
        out_root = read_output_directory().strip()
        if not out_root:
            QMessageBox.warning(
                self,
                "Save all crops",
                "Set an output directory in Settings (Save).",
            )
            return

        fmt = read_save_format()
        root = Path(out_root)
        crops_dir = root / "crops"
        try:
            crops_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Save all crops", str(e))
            return

        r_on, g_on, b_on = self._channel_mask()
        paths = list(self._image_paths)
        n = len(paths)
        prog = QProgressDialog("Saving…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)

        saved = 0
        load_fail: list[str] = []
        skip_no_contour = 0
        skip_bad_patch = 0
        skip_edge = 0
        write_fail: list[str] = []

        for i, p in enumerate(paths):
            prog.setValue(i)
            if prog.wasCanceled():
                break
            prog.setLabelText(f"Saving {p.name}…")
            QApplication.processEvents()

            bgr = load_image_bgr(str(p))
            if bgr is None:
                load_fail.append(p.name)
                continue

            try:
                ok, err = export_crop_to_disk(bgr, p.stem, crops_dir, r_on, g_on, b_on, fmt)
            except OSError as e:
                write_fail.append(f"{p.name}: {e}")
                continue
            if ok:
                saved += 1
            elif err == "no_contour":
                skip_no_contour += 1
            elif err == "edge_bbox":
                skip_edge += 1
            else:
                skip_bad_patch += 1

        prog.setValue(n)

        lines: list[str] = [f"Saved: {saved} ({fmt})"]
        if skip_no_contour:
            lines.append(f"Skipped (no contour): {skip_no_contour}")
        if skip_edge:
            lines.append(f"Skipped (bbox touches edge): {skip_edge}")
        if skip_bad_patch:
            lines.append(f"Skipped (crop too small or degenerate): {skip_bad_patch}")
        if load_fail:
            shown = load_fail[:12]
            extra = f" (+{len(load_fail) - len(shown)} more)" if len(load_fail) > len(shown) else ""
            lines.append(f"Load failed ({len(load_fail)}): {', '.join(shown)}{extra}")
        if write_fail:
            lines.append("Write errors:")
            lines.extend(write_fail[:8])
            if len(write_fail) > 8:
                lines.append(f"(+{len(write_fail) - 8} more)")

        QMessageBox.information(self, "Save all crops", "\n".join(lines))

    def _create_dataset(self) -> None:
        """Create a dataset."""
        dlg = CreateDatasetDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.get_config()
        n = cfg["train_size"] + cfg["test_size"]
        fmt = cfg["label_format"]
        prog = QProgressDialog("Generating dataset…", "Cancel", 0, max(1, n), self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        def cancel_check() -> bool:
            return prog.wasCanceled()

        def progress(step: int, total: int) -> None:
            prog.setMaximum(max(1, total))
            prog.setValue(min(step, prog.maximum()))
            prog.setLabelText(f"Sample {min(step + 1, total)} / {total}")
            QApplication.processEvents()

        stats = run_dataset_generation(
            cfg["crops_dir"],
            cfg["backgrounds_dir"],
            cfg["output_dir"],
            cfg["scale"],
            cfg["train_size"],
            cfg["test_size"],
            fmt,
            output_width=cfg["output_width"],
            output_height=cfg["output_height"],
            label_stem_delimiter=cfg["label_stem_delimiter"],
            crops_per_image=cfg["crops_per_image"],
            max_overlap_percent=cfg["max_overlap_percent"],
            cancel_check=cancel_check,
            progress=progress,
        )
        prog.setValue(prog.maximum())

        lines = [
            f"Saved: {stats.saved}",
            f"Skipped (could not load image): {stats.skipped_load}",
            f"Skipped (could not place crop): {stats.skipped_place}",
        ]
        if stats.errors:
            lines.append("Errors:")
            lines.extend(stats.errors[:20])
            if len(stats.errors) > 20:
                lines.append(f"(+{len(stats.errors) - 20} more)")
        QMessageBox.information(self, "Create dataset", "\n".join(lines))
        if cfg.get("open_when_done"):
            url = QUrl.fromLocalFile(str(cfg["output_dir"].resolve()))
            QDesktopServices.openUrl(url)

    def _set_label_array(self, label: QLabel, arr: np.ndarray | None) -> None:
        """Set the label array for the image viewer."""
        if arr is None or arr.size == 0:
            label.clear()
            label.setText("No image")
            return
        if arr.ndim == 3 and arr.shape[2] == 4:
            qimg = bgra_to_qimage(arr)
        else:
            qimg = bgr_to_qimage(arr)
        pix = QPixmap.fromImage(qimg)
        w = max(1, label.width())
        h = max(1, label.height())
        label.setPixmap(
            pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        label.setText("")

    def _view_array_for_mode(
        self,
        mode: str,
        src: np.ndarray,
        adjusted: np.ndarray,
        mask: np.ndarray | None,
        cnt: np.ndarray | None,
        r_on: bool,
        g_on: bool,
        b_on: bool,
    ) -> np.ndarray:
        """Controls what is displayed in the image viewer."""
        h, w = adjusted.shape[:2]
        black = np.zeros((h, w, 3), dtype=np.uint8)
        if mode == VIEW_ORIGINAL:
            return src
        if mode == VIEW_ADJUSTED:
            return adjusted
        if mode == VIEW_CONTOUR_MASK:
            if mask is None:
                return black
            return cv2.merge([mask, mask, mask])
        if mode == VIEW_CONTOUR_LINES:
            out = adjusted.copy()
            if cnt is not None:
                cv2.drawContours(
                    out, [cnt], -1, (0, 255, 0), thickness=OVERLAY_LINE_THICKNESS
                )
            return out
        if mode == VIEW_BBOX_OVERLAY:
            bl = src.copy()
            if cnt is not None and self._bbox_xywh is not None:
                t = OVERLAY_LINE_THICKNESS
                if self._obb_points_px is not None and self._save_format == FORMAT_YOLO_OBB:
                    pts = self._obb_points_px.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(bl, [pts], True, (0, 255, 0), t)
                else:
                    bx, by, bw, bh = self._bbox_xywh
                    cv2.rectangle(bl, (bx, by), (bx + bw, by + bh), (0, 255, 0), t)
            return bl
        if mode == VIEW_BBOX_CONTENTS:
            if self._image_bgr is None:
                return black
            crop, err = compute_export_crop(
                self._image_bgr, r_on, g_on, b_on, self._save_format
            )
            if err is not None or crop is None:
                return black
            return crop
        return adjusted

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._load_thread.quit()
        self._load_thread.wait(10_000)
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        """Update the views when the window is resized."""
        super().resizeEvent(event)
        if self._image_bgr is not None:
            self._update_views()

    def _update_views(self) -> None:
        """Update the views."""
        if self._image_bgr is None:
            for lab in (self._label_tl, self._label_tr, self._label_bl, self._label_br):
                lab.clear()
                lab.setText("No image")
            self._last_largest_contour = None
            self._bbox_xywh = None
            self._obb_points_px = None
            self._export_w = self._export_h = 0
            return

        src = self._image_bgr
        r_on, g_on, b_on = self._channel_mask()
        adjusted = apply_channel_mask(src, r_on, g_on, b_on)

        # Single-channel input for Otsu: mean of B, G, R (no LAB/L channel, no BGR2GRAY).
        gray = np.mean(adjusted, axis=2).astype(np.uint8)
        mask, cnt = largest_contour_mask(gray)

        self._export_h, self._export_w = adjusted.shape[:2]

        self._last_largest_contour = cnt
        self._bbox_xywh = None
        self._obb_points_px = None
        if cnt is not None:
            self._bbox_xywh = axis_aligned_bbox_from_contour(cnt)
            self._obb_points_px = oriented_box_points(cnt)

        labels = (self._label_tl, self._label_tr, self._label_bl, self._label_br)
        for i, lab in enumerate(labels):
            mode = self._view_slots[i] if i < len(self._view_slots) else DEFAULT_VIEW_SLOTS[i]
            arr = self._view_array_for_mode(
                mode, src, adjusted, mask, cnt, r_on, g_on, b_on
            )
            self._set_label_array(lab, arr)


def main() -> None:
    app = QApplication([])
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
