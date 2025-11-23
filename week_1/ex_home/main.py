import os
import cv2
from utils import (
    list_images, ensure_output_dirs, save_to_formats,
    show_each_image, split_bgr_to_rgb_imgs, to_gray,
    rotate_sequence_same_window, crop_center
)

# ----------------- Paths -----------------
# Chạy file này từ week_1/ex_home/
IMAGES_DIR = os.path.normpath(os.path.join("..", "images"))
OUT_ROOT   = "."  # outputs sẽ nằm trong ./outputs/

def part_1_save_formats(img_paths):
    print("[HW1] Save each image to PNG/JPG (normalized to 1200x627 landscape) ...")
    out_dirs = ensure_output_dirs(OUT_ROOT)
    for p in img_paths:
        saved = save_to_formats(p, out_dirs)
        print("  -", os.path.basename(p), "->", saved)
    print("Done.\n")

def part_2_show_each_window(img_paths):
    print("[HW2.1] Show each image on each window (press any key to go next, q/Esc to quit)")
    show_each_image(img_paths, delay_mode="manual")
    cv2.destroyAllWindows()
    print("Done.\n")

def part_3_split_rgb_and_show(sample_img_path):
    print("[HW2.2] Separate RGB to 3 layers and show each on each window")
    img = cv2.imread(sample_img_path)
    if img is None:
        print("  [WARN] Cannot read image:", sample_img_path)
        return
    R_img, G_img, B_img = split_bgr_to_rgb_imgs(img)
    cv2.imshow("Layer R (red only)", R_img)
    cv2.imshow("Layer G (green only)", G_img)
    cv2.imshow("Layer B (blue only)", B_img)
    print("  -> Press any key to continue ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done.\n")

def part_4_to_gray(sample_img_path):
    print("[HW2.3] Convert RGB image to Gray and show it")
    img = cv2.imread(sample_img_path)
    if img is None:
        print("  [WARN] Cannot read image:", sample_img_path)
        return
    gray = to_gray(img)
    cv2.imshow("Gray image", gray)
    print("  -> Press any key to continue ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done.\n")

def part_5_rotate_100_steps(sample_img_path):
    print("[HW2.4] Rotate an image 100 times (+5° each, 0.1s pause), show on SAME window")
    img = cv2.imread(sample_img_path)
    if img is None:
        print("  [WARN] Cannot read image:", sample_img_path)
        return
    rotate_sequence_same_window(
        img, steps=100, angle_step=5.0,
        window_name="Rotate Animation", delay_ms=100
    )
    cv2.destroyAllWindows()
    print("Done.\n")

def part_6_crop_center(sample_img_path):
    print("[HW2.5] Crop centered region with 1/4 size (W/2 x H/2) and show it")
    img = cv2.imread(sample_img_path)
    if img is None:
        print("  [WARN] Cannot read image:", sample_img_path)
        return
    cropped = crop_center(img, frac=0.5)
    cv2.imshow("Cropped center (W/2 x H/2)", cropped)
    print("  -> Press any key to finish ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done.\n")

def main():
    # 0) lấy danh sách ảnh nguồn
    img_paths = list_images(IMAGES_DIR)

    # 1) Lưu ra 2 định dạng PNG/JPG (chuẩn hoá 1200x627 landscape)
    part_1_save_formats(img_paths)

    # 2) Hiển thị mỗi ảnh trên một cửa sổ
    part_2_show_each_window(img_paths)

    # Chọn 1 ảnh tiêu biểu để làm các phần còn lại (dùng ảnh đầu tiên)
    sample = img_paths[0]

    # 3) Tách 3 kênh và hiển thị
    part_3_split_rgb_and_show(sample)

    # 4) Chuyển xám và hiển thị
    part_4_to_gray(sample)

    # 5) Quay 100 bước (trên cùng 1 cửa sổ, mỗi bước +5°, pause 0.1s)
    part_5_rotate_100_steps(sample)

    # 6) Crop từ tâm kích thước 1/4 theo cạnh
    part_6_crop_center(sample)

if __name__ == "__main__":
    main()
