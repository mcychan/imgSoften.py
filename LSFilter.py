import numpy as np

# Wang, H., Cao, J., Liu, X. et al. Least-squares images for edge-preserving smoothing.
# Comp. Visual Media 1, 27–35 (2015).
# https://doi.org/10.1007/s41095-015-0004-6
# Copyright (c) 2023 Miller Cy Chan

class LSFilter:
    def __init__(self, pixels, side = 5, beta = 1e-6, lamda = 5):
        self._width, self._height, _ = pixels.shape
        self._pixels = pixels
        self._side = np.clip(side, 3, 5)
        self._offset = self._side // 2
        self._beta = beta
        self._lamda = lamda


    def getValue(self, x, y, x1, y1):
        i, j = x + y * self._width, x1 + y1 * self._width
        if j < 0 or x1 >= self._width or y1 >= self._height:
            return 0
        if i == j:
            return 1

        pixels = self._pixels
        value = np.sum(np.abs(pixels[x1, y1] - pixels[x, y]))
        return -np.exp(-self._beta * value)


    def getSmoothedKernel(self, x, y):
        offset, side = self._offset, self._side
        I = np.eye(side)
        laplacian = np.zeros(I.shape)
        for j in range(side):
            y1 = y + j - offset
            for i in range(side):
                x1 = x + i - offset
                laplacian[i, j] = self.getValue(x, y, x1, y1)

        result = I * self._lamda + laplacian @ laplacian.T
        return np.linalg.inv(result) * self._lamda


    def doFilter(self, x, y):
        offset, pixels, side = self._offset, self._pixels, self._side
        width, height = self._width, self._height
        sMatrix = self.getSmoothedKernel(x, y)

        acc = np.zeros(pixels.shape[2])
        for j in range(side):
            y1 = y + j - offset
            if y1 < 0:
                continue
            if y1 >= height:
                break
            for i in range(side):
                x1 = x + j - offset
                if x1 < 0:
                    continue
                if x1 >= width:
                    break
                acc += pixels[x1, y1] * sMatrix[i, j]
        return np.clip(acc, 0, 255)


    def filter(self):
        pixels, qPixels = self._pixels, np.copy(self._pixels)
        width, height = self._width, self._height
        for y in range(height):
            for x in range(width):
                qPixels[x, y] = self.doFilter(x, y)
        return qPixels
