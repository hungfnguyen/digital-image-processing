import cv2
import numpy as np

def equalize_histogram(image):
    """Cân bằng Histogram"""
    if len(image.shape) == 2:
        # Ảnh xám
        return cv2.equalizeHist(image)
    else:
        # Ảnh màu: chuyển sang YCrCb để equalize kênh Y
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def plot_histogram(image):
    """Trả về histogram (256 giá trị) để vẽ nếu cần"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    return hist
