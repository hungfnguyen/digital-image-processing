# frequency.py
import cv2
import numpy as np
import time

def dft_transform(image):
    """Chuyển ảnh sang miền tần số (DFT)."""
    img_float = np.float32(image)
    # DFT số phức
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    # Đưa tần số thấp về tâm
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

def idft_transform(dft_shift):
    """Chuyển ngược về miền không gian (IDFT)."""
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    # Tính độ lớn (magnitude)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    # Chuẩn hóa về 0-255 để hiển thị
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_back)

def create_filter_mask(shape, d0, filter_type, n=2):
    """Tạo mặt nạ bộ lọc H(u,v)."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Tạo lưới tọa độ
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    # Tính khoảng cách D từ tâm
    D = np.sqrt((V - crow)**2 + (U - ccol)**2)
    
    mask = np.zeros((rows, cols, 2), np.float32)
    
    # --- IDEAL ---
    if filter_type == 'ideal_lp':
        h = np.where(D <= d0, 1, 0)
    elif filter_type == 'ideal_hp':
        h = np.where(D > d0, 1, 0)
        
    # --- BUTTERWORTH ---
    elif filter_type == 'butter_lp':
        h = 1 / (1 + (D / d0)**(2 * n))
    elif filter_type == 'butter_hp':
        with np.errstate(divide='ignore', invalid='ignore'):
            h = 1 / (1 + (d0 / (D + 1e-5))**(2 * n))
            
    # --- GAUSSIAN ---
    elif filter_type == 'gauss_lp':
        h = np.exp(-(D**2) / (2 * d0**2))
    elif filter_type == 'gauss_hp':
        h = 1 - np.exp(-(D**2) / (2 * d0**2))
    else:
        return None

    mask[:,:,0] = h
    mask[:,:,1] = h
    return mask

def apply_filter(image, d0=30, filter_type='ideal_lp', n=2):
    """Hàm gọi chung cho các slider."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    dft_shift = dft_transform(gray)
    mask = create_filter_mask(gray.shape, d0, filter_type, n)
    fshift_filtered = dft_shift * mask
    img_back = idft_transform(fshift_filtered)
    
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

# ---------------------------------------------------------
# CÁC HÀM BÀI TẬP (HOMEWORK 3 & BENCHMARK)
# ---------------------------------------------------------

def get_gaussian_kernel(shape, d0, highpass=False):
    """Hàm phụ trợ tạo kernel Gaussian nhanh."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v, u)
    D2 = (V - crow)**2 + (U - ccol)**2
    H = np.exp(-D2 / (2 * d0**2))
    if highpass:
        H = 1 - H
    return H

def apply_filter_sequence(image, d0=25):
    """HW3-1: Lowpass -> Highpass tuần tự."""
    start_time = time.time()
    
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image

    dft_shift = dft_transform(gray)
    
    # Tạo 2 filter
    H_low = get_gaussian_kernel(gray.shape, d0, highpass=False)
    H_high = get_gaussian_kernel(gray.shape, d0, highpass=True)
    
    # Nhân dồn: F_new = F * H_low * H_high
    mask_low = np.dstack([H_low, H_low])
    mask_high = np.dstack([H_high, H_high])
    
    fshift_filtered = dft_shift * mask_low * mask_high
    img_out = idft_transform(fshift_filtered)
    
    exec_time = (time.time() - start_time) * 1000
    return cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR), exec_time

def apply_multi_pass(image, d0=30, passes=1):
    """HW3-2: Chạy Highpass n lần."""
    start_time = time.time()
    
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
        
    dft_shift = dft_transform(gray)
    H = get_gaussian_kernel(gray.shape, d0, highpass=True)
    
    # Lũy thừa bộ lọc: H^n
    H_final = np.power(H, passes)
    
    mask = np.dstack([H_final, H_final])
    fshift_filtered = dft_shift * mask
    img_out = idft_transform(fshift_filtered)
    
    exec_time = (time.time() - start_time) * 1000
    return cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR), exec_time

def benchmark_spatial_vs_freq(image, ksize=15):
    """So sánh tốc độ."""
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # 1. Spatial
    t0 = time.time()
    cv2.GaussianBlur(image, (ksize, ksize), 0)
    t_spatial = (time.time() - t0) * 1000
    
    # 2. Frequency
    t1 = time.time()
    dft_shift = dft_transform(image)
    H = get_gaussian_kernel(image.shape, d0=30, highpass=False) 
    mask = np.dstack([H, H])
    idft_transform(dft_shift * mask)
    t_freq = (time.time() - t1) * 1000
    
    return t_spatial, t_freq