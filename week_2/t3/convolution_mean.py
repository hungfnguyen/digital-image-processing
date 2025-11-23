# convolution_mean.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Viết hàm convolution 2D
# ===============================
def conv(A, k, padding=True):
    kh, kw = k.shape
    h, w = A.shape

    # Padding để không mất biên
    if padding:
        pad_h, pad_w = kh // 2, kw // 2
        A = np.pad(A, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    h2, w2 = A.shape
    C = np.zeros((h2 - kh + 1, w2 - kw + 1))

    # Tích chập thủ công
    for i in range(h2 - kh + 1):
        for j in range(w2 - kw + 1):
            region = A[i:i+kh, j:j+kw]
            C[i, j] = np.sum(region * k)

    return C

# ===============================
# 2. Đọc ảnh
# ===============================
img = cv2.imread('images/fox.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (200, 200))

# ===============================
# 3. Kernel 5x5 – Mean Filter
# ===============================
kernel = np.ones((5, 5)) / 25

# ===============================
# 4. Áp dụng cho từng kênh RGB
# ===============================
r, g, b = cv2.split(img)
R = conv(r, kernel)
G = conv(g, kernel)
B = conv(b, kernel)

img_out = cv2.merge((R, G, B))
img_out = np.clip(img_out, 0, 255).astype(np.uint8)

# ===============================
# 5. Hiển thị kết quả
# ===============================
plt.figure(dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_out)
plt.title("Mean Filter 5x5")
plt.axis("off")

plt.show()
