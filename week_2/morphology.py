import cv2
import numpy as np

def get_kernel(size):
    if size < 1: size = 1
    if size % 2 == 0: size += 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

def erode(img, size):
    kernel = get_kernel(size)
    return cv2.erode(img, kernel, iterations=1)

def dilate(img, size):
    kernel = get_kernel(size)
    return cv2.dilate(img, kernel, iterations=1)

def opening(img, size):
    kernel = get_kernel(size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img, size):
    kernel = get_kernel(size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def gradient(img, size):
    kernel = get_kernel(size)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def boundary_extraction(img, size):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    kernel = get_kernel(size)
    eroded = cv2.erode(gray, kernel, iterations=1)
    boundary = cv2.subtract(gray, eroded)
    return cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)

def fill_holes(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape[:2]
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    im_floodfill = padded.copy()
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out_padded = padded | im_floodfill_inv
    im_out = im_out_padded[1:-1, 1:-1]
    
    if len(img.shape) == 3:
        return cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)
    return im_out

def analyze_objects_with_filling(img):

    im_filled = fill_holes(img)
    if len(im_filled.shape) == 3:
        im_gray = cv2.cvtColor(im_filled, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im_filled

    kernel_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    im_separated = cv2.morphologyEx(im_gray, cv2.MORPH_OPEN, kernel_sep) 

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im_separated, connectivity=8)
    
    n_objects = num_labels - 1
    output_img = cv2.cvtColor(im_separated, cv2.COLOR_GRAY2BGR)
    results_text = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50: 
            n_objects -= 1
            continue

        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(output_img, (cx, cy), 4, (0, 0, 255), -1)
        x, y, w, h, _ = stats[i]
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(output_img, f"{area}", (cx - 20, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        results_text.append(f"Obj{i}: {area}")
    return output_img, n_objects, results_text

def clean_fingerprint(img, kernel_size=3):
    """
    Làm sạch ảnh vân tay: Opening (xóa nhiễu trắng) -> Closing (hàn gắn vết đứt).
    """
    # 1. Chuẩn hóa ảnh về Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # 2. Nhị phân hóa (Thresholding)
    # Dùng Otsu để tự động tìm ngưỡng tối ưu, hoặc fix cứng 127
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Tạo Structuring Element (Kernel)
    # Kích thước kernel rất quan trọng:
    # - Quá nhỏ: Không xóa hết nhiễu.
    # - Quá lớn: Làm mất chi tiết vân tay.
    kernel = get_kernel(kernel_size)
    
    # 4. Bước 1: Opening (Co -> Giãn) để xóa nhiễu hạt trắng
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 5. Bước 2: Closing (Giãn -> Co) để lấp lỗ đen/nối liền vân
    # Áp dụng trên kết quả của bước Opening
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Chuyển về 3 kênh màu để hiển thị trên GUI
    if len(img.shape) == 3:
        return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    return closed