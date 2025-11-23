import os
from typing import List, Dict, Tuple
import cv2
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------- Files & folders ----------
def list_images(dir_path: str) -> List[str]:
    """Return sorted list of image absolute paths in dir_path."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Folder not found: {dir_path}")
    files = []
    for name in sorted(os.listdir(dir_path)):
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXT:
            files.append(os.path.join(dir_path, name))
    if not files:
        raise RuntimeError(f"No images found in: {dir_path}")
    return files

def ensure_output_dirs(root: str) -> Dict[str, str]:
    """Create outputs/{png,jpg} and return mapping ext->dir."""
    sub = {
        "png": os.path.join(root, "outputs", "png"),
        "jpg": os.path.join(root, "outputs", "jpg"),
    }
    for p in sub.values():
        os.makedirs(p, exist_ok=True)
    return sub

# ---------- Image helpers ----------
TARGET_SIZE = (1200, 627)  # (W, H) per HW1

def to_landscape_1200x627(img: np.ndarray) -> np.ndarray:
    """
    Chuẩn hoá ảnh theo yêu cầu HW1:
    - Đảm bảo landscape (nếu cao > rộng thì xoay 90°).
    - Resize về đúng 1200x627.
    """
    h, w = img.shape[:2]
    if h > w:  # portrait -> rotate to landscape
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # resize to target
    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized

def save_to_formats(img_path: str, out_dirs: Dict[str, str]) -> Dict[str, str]:
    """
    Save the given image to PNG/JPG in outputs (after normalizing to 1200x627 landscape).
    Return ext->saved_path.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    # nếu ảnh là gray => chuyển sang BGR để lưu đồng nhất
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_norm = to_landscape_1200x627(img)

    base = os.path.splitext(os.path.basename(img_path))[0]
    saved = {}
    # PNG
    png_path = os.path.join(out_dirs["png"], f"{base}.png")
    cv2.imwrite(png_path, img_norm)
    saved["png"] = png_path
    # JPG (quality 95)
    jpg_path = os.path.join(out_dirs["jpg"], f"{base}.jpg")
    cv2.imwrite(jpg_path, img_norm, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved["jpg"] = jpg_path
    return saved

# ---------- Display helpers ----------
def show_each_image(paths: List[str], delay_mode="manual", delay_ms=400):
    """
    Show each image in its own window.
    delay_mode: 'manual' => bấm phím để qua ảnh; 'auto' => tự chạy delay_ms.
    """
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Skip unreadable: {p}")
            continue
        win = f"Image_{i:02d}"
        cv2.imshow(win, img)
        key = cv2.waitKey(0 if delay_mode == "manual" else delay_ms) & 0xFF
        if key in (27, ord('q')):
            break

def split_bgr_to_rgb_imgs(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OpenCV đọc BGR. Trả về 3 ảnh màu nhấn mạnh từng kênh R/G/B.
    """
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)
    R_img = cv2.merge([zeros, zeros, r])
    G_img = cv2.merge([zeros, g, zeros])
    B_img = cv2.merge([b, zeros, zeros])
    return R_img, G_img, B_img

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- Geometric transforms ----------
def rotate_sequence_same_window(
    img: np.ndarray,
    steps: int = 100,
    angle_step: float = 5.0,
    window_name: str = "Rotate Animation",
    delay_ms: int = 100,
):
    """
    Animation trên CÙNG MỘT CỬA SỔ:
    - Bước i: angle = i * angle_step
    - KHÔNG scale (scale = 1.0)
    - ESC / 'q' để thoát sớm
    """
    (h, w) = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    for i in range(steps):
        angle = i * angle_step
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord('q')):  # ESC hoặc 'q'
            break

def crop_center(img: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """
    Crop từ tâm; frac=0.5 => lấy W*0.5 x H*0.5 (¼ theo cạnh).
    """
    h, w = img.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - cw // 2)
    y0 = max(0, cy - ch // 2)
    x1 = min(w, x0 + cw)
    y1 = min(h, y0 + ch)
    return img[y0:y1, x0:x1].copy()
