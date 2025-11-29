import cv2
import numpy as np

# dùng một ngưỡng cố định
def global_threshold(img, thresh_val=127):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# chia anh thành các vùng nhỏ  độ sáng trung bình
def adaptive_threshold(img, method='mean', block_size=11, C=2):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if block_size % 2 == 0: block_size += 1
    if block_size < 3: block_size = 3

    # trung bình có trọng số
    if method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

    binary = cv2.adaptiveThreshold(
        gray, 255, adaptive_method, 
        cv2.THRESH_BINARY, block_size, C
    )
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def otsu_threshold(img):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"Otsu found threshold: {thresh_val}")
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def canny_edge(img, t_lower=100, t_upper=200):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    edges = cv2.Canny(gray, t_lower, t_upper)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def hough_lines(img, t_lower=50, t_upper=150, threshold=100):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = img.copy()
    else:
        gray = img
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    edges = cv2.Canny(gray, t_lower, t_upper)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    return output

def hough_circles(img, min_dist=20, param1=50, param2=30, min_r=0, max_r=0):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = img.copy()
    else:
        gray = img
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray_blurred = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
            
    return output

def watershed_segmentation(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_color = img.copy()
    else:
        gray = img
        original_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(original_color, markers)
    original_color[markers == -1] = [0, 0, 255]

    return original_color