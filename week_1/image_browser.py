import os
import cv2
import numpy as np

# ---------- Config ----------
IMAGE_DIR = "images"                 # thư mục chứa ảnh
PLACEHOLDER_PATH = "/home/hungfnguyen/Documents/digital-image-processing/no_image.png"
WIN_NAME  = "OpenCV Image Browser"

# Panel trái (grid thumbnails)
THUMB_W, THUMB_H = 110, 80           # kích thước thumbnail
GRID_COLS = 4                         # số cột trong panel trái
GAP_X, GAP_Y = 12, 16                 # khoảng cách ngang/dọc giữa thumbnail
PAD_X, PAD_Y = 10, 10                 # lề panel trái

# Panel phải (ảnh lớn)
RIGHT_MIN_W = 720                     # độ rộng tối thiểu panel phải
HEIGHT   = 720                        # chiều cao cửa sổ tổng

# Màu nền
BG_LEFT  = (32, 32, 32)
BG_RIGHT = (24, 24, 24)
ACCENT   = (90, 180, 255)             # viền khi chọn

# ---------- Load file list ----------
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
files = []
if os.path.isdir(IMAGE_DIR):
    for name in sorted(os.listdir(IMAGE_DIR)):
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXT:
            files.append(os.path.join(IMAGE_DIR, name))
else:
    raise SystemExit(f"Folder '{IMAGE_DIR}' không tồn tại.")

# ---------- Preload placeholder ----------
placeholder = cv2.imread(PLACEHOLDER_PATH)
if placeholder is None:
    # fallback: khung xám nếu thiếu ảnh placeholder
    placeholder = np.full((300, 500, 3), 120, np.uint8)

# ---------- Prepare thumbnails ----------
thumbs = []
for path in files:
    img = cv2.imread(path)
    if img is None:
        thumbs.append(None)
        continue
    h, w = img.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    t = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    y0 = (THUMB_H - new_h) // 2
    x0 = (THUMB_W - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = t
    thumbs.append(canvas)

# ---------- Layout ----------
left_panel_w  = PAD_X * 2 + GRID_COLS * THUMB_W + (GRID_COLS - 1) * GAP_X
right_panel_w = RIGHT_MIN_W
WINDOW_W      = left_panel_w + right_panel_w
visible_rows  = max(1, (HEIGHT - PAD_Y * 2 + GAP_Y) // (THUMB_H + GAP_Y))

selected_idx = None           # <<< KHÔNG chọn ảnh nào khi khởi động
scroll_row = 0                # cuộn theo hàng

def clamp_scroll():
    global scroll_row
    total_rows = (len(files) + GRID_COLS - 1) // GRID_COLS
    max_top = max(0, total_rows - visible_rows)
    if scroll_row < 0:
        scroll_row = 0
    elif scroll_row > max_top:
        scroll_row = max_top

def ensure_visible(idx: int):
    """Đưa hàng của idx vào vùng hiển thị."""
    global scroll_row
    row_of_idx = idx // GRID_COLS
    if row_of_idx < scroll_row:
        scroll_row = row_of_idx
    elif row_of_idx >= scroll_row + visible_rows:
        scroll_row = row_of_idx - visible_rows + 1
    clamp_scroll()

def fit_on_canvas(img, canvas_w, canvas_h, bg):
    if img is None:
        return np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    y0 = (canvas_h - new_h) // 2
    x0 = (canvas_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

# Cache ảnh gốc khi đã mở
full_cache = {}

def draw_left_panel():
    left = np.full((HEIGHT, left_panel_w, 3), BG_LEFT, dtype=np.uint8)
    y = PAD_Y
    start_idx = scroll_row * GRID_COLS
    end_idx = min(len(files), start_idx + visible_rows * GRID_COLS)

    idx = start_idx
    for _ in range(visible_rows):
        x = PAD_X
        for _ in range(GRID_COLS):
            if idx >= end_idx:
                break
            if thumbs[idx] is not None:
                left[y:y+THUMB_H, x:x+THUMB_W] = thumbs[idx]
            else:
                cv2.rectangle(left, (x, y), (x+THUMB_W, y+THUMB_H), (80, 80, 80), 1)
                cv2.line(left, (x, y), (x+THUMB_W, y+THUMB_H), (80, 80, 80), 1)
                cv2.line(left, (x+THUMB_W, y), (x, y+THUMB_H), (80, 80, 80), 1)
            if selected_idx is not None and idx == selected_idx:
                cv2.rectangle(left, (x-4, y-4), (x+THUMB_W+4, y+THUMB_H+4), ACCENT, 2)
            x += THUMB_W + GAP_X
            idx += 1
        y += THUMB_H + GAP_Y
        if idx >= end_idx:
            break
    return left

def draw_right_panel():
    """Nếu chưa chọn ảnh, hiển thị placeholder."""
    if selected_idx is None:
        return fit_on_canvas(placeholder, right_panel_w, HEIGHT, BG_RIGHT)

    path = files[selected_idx]
    if path in full_cache:
        img = full_cache[path]
    else:
        img = cv2.imread(path)
        full_cache[path] = img
    right = fit_on_canvas(img, right_panel_w, HEIGHT, BG_RIGHT)

    # thanh tiêu đề đen + tên file
    base = os.path.basename(path)
    cv2.rectangle(right, (0, 0), (right_panel_w, 28), (0, 0, 0), -1)
    cv2.putText(right, f"{selected_idx+1}/{len(files)}  {base}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return right

def render():
    frame = np.hstack([draw_left_panel(), draw_right_panel()])
    cv2.imshow(WIN_NAME, frame)

def select_first_if_none():
    global selected_idx
    if selected_idx is None and len(files) > 0:
        selected_idx = 0
        ensure_visible(selected_idx)

# ---------- Mouse & Keyboard ----------
def mouse_cb(event, x, y, flags, userdata):
    global selected_idx, scroll_row
    if event == cv2.EVENT_LBUTTONDOWN and x < left_panel_w:
        inside_x = x - PAD_X
        inside_y = y - PAD_Y
        if inside_x >= 0 and inside_y >= 0:
            cell_w = THUMB_W + GAP_X
            cell_h = THUMB_H + GAP_Y
            col = int(inside_x // cell_w)
            row = int(inside_y // cell_h)
            if 0 <= col < GRID_COLS and 0 <= row < visible_rows:
                idx = (scroll_row + row) * GRID_COLS + col
                if 0 <= idx < len(files):
                    selected_idx = idx
                    ensure_visible(selected_idx)
                    render()

    if event == cv2.EVENT_MOUSEWHEEL:
        delta = 1 if flags > 0 else -1
        scroll_row -= delta
        clamp_scroll()
        render()

cv2.namedWindow(WIN_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(WIN_NAME, WINDOW_W, HEIGHT)
cv2.setMouseCallback(WIN_NAME, mouse_cb)

render()

while True:
    key = cv2.waitKey(20) & 0xFFFF
    if key in (27, ord('q')):  # Esc hoặc q
        break
    elif key == 0x260000:  # ↑ : lên 1 hàng
        if selected_idx is None:
            select_first_if_none()
        elif selected_idx - GRID_COLS >= 0:
            selected_idx -= GRID_COLS
            ensure_visible(selected_idx)
        render()
    elif key == 0x280000:  # ↓ : xuống 1 hàng
        if selected_idx is None:
            select_first_if_none()
        elif selected_idx + GRID_COLS < len(files):
            selected_idx += GRID_COLS
            ensure_visible(selected_idx)
        render()
    elif key == 0x250000:  # ← : sang trái
        if selected_idx is None:
            select_first_if_none()
        elif (selected_idx % GRID_COLS) > 0:
            selected_idx -= 1
            ensure_visible(selected_idx)
        render()
    elif key == 0x270000:  # → : sang phải
        if selected_idx is None:
            select_first_if_none()
        elif (selected_idx % GRID_COLS) < GRID_COLS - 1 and selected_idx + 1 < len(files):
            selected_idx += 1
            ensure_visible(selected_idx)
        render()
    elif key == 0x210000:  # PageUp
        if selected_idx is None:
            select_first_if_none()
        else:
            selected_idx = max(0, selected_idx - GRID_COLS * visible_rows)
            ensure_visible(selected_idx)
        render()
    elif key == 0x220000:  # PageDown
        if selected_idx is None:
            select_first_if_none()
        else:
            selected_idx = min(len(files) - 1, selected_idx + GRID_COLS * visible_rows)
            ensure_visible(selected_idx)
        render()
    elif key == 0x240000:  # Home
        if len(files) > 0:
            selected_idx = 0
            ensure_visible(selected_idx)
            render()
    elif key == 0x230000:  # End
        if len(files) > 0:
            selected_idx = len(files) - 1
            ensure_visible(selected_idx)
            render()

cv2.destroyAllWindows()
