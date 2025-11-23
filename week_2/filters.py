import cv2
import numpy as np

def mean_filter(image, ksize=3):
    """Lọc trung bình (Mean Filter)"""
    return cv2.blur(image, (ksize, ksize))

def gaussian_filter(image, ksize=3, sigma=1):
    """Lọc Gaussian (Low-pass)"""
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def median_filter(image, ksize=3):
    """Lọc trung vị (Median Filter)"""
    return cv2.medianBlur(image, ksize)

def laplacian_filter(image):
    """Lọc Laplacian (High-pass)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def sobel_filter(image):
    """Lọc Sobel (High-pass edge detection)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
