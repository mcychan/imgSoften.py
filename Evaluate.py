import cv2
import numpy as np


def calculate_psnr(src, dst, row = 0, col = 0):
    h, w = src.shape[: 2]
    e = cv2.absdiff(src.astype(dtype = np.float32), dst.astype(dtype = np.float32))
    e = e[row : h - row, col : w - col]
    mse = np.mean(e ** 2)
    if mse == 0:
        return np.INF
    return 10 * np.log10((255 ** 2) / mse)


def __ssim(src, dst, row = 0, col = 0):
    h, w = src.shape[: 2]
    k1, k2, L = 0.01, 0.03, 255
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)

    src1, dst1 = src[row : h - row, col : w - col].astype(dtype = np.float32), dst[row : h - row, col : w - col].astype(dtype = np.float32)
    mu1, mu2 = cv2.filter2D(src1, -1, window)[5 : -5, 5 : -5], cv2.filter2D(dst1, -1, window)[5 : -5, 5 : -5]
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(src1 ** 2, -1, window)[5 : -5, 5 : -5] - mu1_sq
    sigma2_sq = cv2.filter2D(dst1 ** 2, -1, window)[5 : -5, 5 : -5] - mu2_sq
    sigma12 = cv2.filter2D(src1 * dst1, -1, window)[5 : -5, 5 : -5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(src, dst, row = 0, col = 0):
    if not src.shape == dst.shape:
        raise ValueError("Input images must have the same dimensions.")
    if src.ndim == 2:
        return __ssim(src, dst, row, col)
    if src.ndim == 3:
        if src.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(__ssim(src, dst, row, col))
            return np.array(ssims).mean()
        if src.shape[2] == 1:
            return __ssim(np.squeeze(src), np.squeeze(dst), row, col)
    else:
        raise ValueError("Wrong input image dimensions.")
