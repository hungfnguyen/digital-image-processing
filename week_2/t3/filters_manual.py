import cv2
import numpy as np
import matplotlib.pyplot as plt

def min_filter(img, ksize=3):
    pad = ksize // 2
    h, w, c = img.shape
    out = np.zeros_like(img)

    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y+ksize, x:x+ksize, ch]
                out[y, x, ch] = np.min(region)
    return out


def max_filter(img, ksize=3):
    pad = ksize // 2
    h, w, c = img.shape
    out = np.zeros_like(img)

    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y+ksize, x:x+ksize, ch]
                out[y, x, ch] = np.max(region)
    return out


def mean_filter(img, ksize=3):
    pad = ksize // 2
    h, w, c = img.shape
    out = np.zeros_like(img, dtype=float)

    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y+ksize, x:x+ksize, ch]
                out[y, x, ch] = np.mean(region)
    return np.clip(out, 0, 255).astype(np.uint8)


def median_filter(img, ksize=3):
    pad = ksize // 2
    h, w, c = img.shape
    out = np.zeros_like(img)

    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y+ksize, x:x+ksize, ch]
                out[y, x, ch] = np.median(region)
    return out


def gaussian_kernel(ksize=5, sigma=1.0):
    center = ksize // 2
    kernel = np.zeros((ksize, ksize), dtype=float)
    for i in range(ksize):
        for j in range(ksize):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def gaussian_filter(img, ksize=5, sigma=1.0):
    pad = ksize // 2
    h, w, c = img.shape
    out = np.zeros_like(img, dtype=float)
    kernel = gaussian_kernel(ksize, sigma)

    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                region = padded[y:y+ksize, x:x+ksize, ch]
                out[y, x, ch] = np.sum(region * kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


img = cv2.imread("images/fox.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (200, 200))

filters = {
    "Original": img,
    "Min Filter": min_filter(img, 3),
    "Max Filter": max_filter(img, 3),
    "Mean Filter": mean_filter(img, 5),
    "Median Filter": median_filter(img, 5),
    "Gaussian Filter": gaussian_filter(img, 5, 1.0)
}

plt.figure(figsize=(10, 6), dpi=150)
for i, (name, fimg) in enumerate(filters.items()):
    plt.subplot(2, 3, i + 1)
    plt.imshow(fimg)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()
