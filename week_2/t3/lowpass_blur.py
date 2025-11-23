# lowpass_blur.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# ===============================
# 1. Hàm convolution dùng scipy
# ===============================
def Conv(img, k):
    if img.ndim == 3:
        out = np.zeros_like(img)
        for i in range(3):
            out[:, :, i] = convolve(img[:, :, i], k)
        return out
    else:
        return convolve(img, k)

# ===============================
# 2. Hàm Gaussian kernel
# ===============================
def Gausskernel(size=5, sigma=1.5):
    s = (size - 1) // 2
    ax = np.linspace(-s, s, size)
    gauss = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

# ===============================
# 3. Đọc ảnh
# ===============================
img = cv2.imread('images/fox.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (200, 200))

# ===============================
# 4. Mean Filter (11x11)
# ===============================
mean_k = np.ones((11, 11)) / (11 * 11)
mean_img = Conv(img, mean_k)

# ===============================
# 5. Gaussian Filter (11x11, σ=3)
# ===============================
gauss_k = Gausskernel(11, 3)
gauss_img = Conv(img, gauss_k)

# ===============================
# 6. Hiển thị kết quả
# ===============================
plt.figure(figsize=(10, 4), dpi=150)
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mean_img.astype(np.uint8))
plt.title("Mean Filter (11x11)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gauss_img.astype(np.uint8))
plt.title("Gaussian Filter (σ=3)")
plt.axis('off')

plt.show()
