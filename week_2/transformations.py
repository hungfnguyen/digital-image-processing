import cv2
import numpy as np

def negative(image):
    """Âm bản"""
    return 255 - image

def log_transform(image, c=1.0):
    """Biến đổi Logarithm: s = c * log(1 + r)"""
    img_float = np.float32(image)
    # Tính s = c * log(1 + r)
    result = c * np.log1p(img_float)
    result = np.clip(result, 0, 255)
    return np.uint8(result)

def gamma_transform(image, gamma=1.0):
    """Biến đổi Gamma: s = c * r^gamma"""
    img_norm = np.float32(image) / 255.0
    result = np.power(img_norm, gamma)
    # Normalize lại về 0-255 để hiển thị đúng
    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
    return np.uint8(result * 255)

def piecewise_linear(image, r1=70, r2=140, s1=0, s2=255):
    """Biến đổi đoạn tuyến tính (Contrast Stretching)"""
    # Bảo vệ lỗi chia cho 0 nếu r1 == r2
    if r1 >= r2:
        r2 = r1 + 1
    
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < r1:
            # Đoạn 1: 0 -> r1
            if r1 == 0: val = 0
            else: val = (s1 / r1) * i
        elif i < r2:
            # Đoạn 2: r1 -> r2
            val = ((s2 - s1) / (r2 - r1)) * (i - r1) + s1
        else:
            # Đoạn 3: r2 -> 255
            if r2 == 255: val = s2
            else: val = ((255 - s2) / (255 - r2)) * (i - r2) + s2
            
        lookup_table[i] = int(np.clip(val, 0, 255))
        
    return cv2.LUT(image, lookup_table)