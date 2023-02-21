import numpy as np

# Wang, H., Cao, J., Liu, X. et al. Least-squares images for edge-preserving smoothing.
# Comp. Visual Media 1, 27–35 (2015).
# https://doi.org/10.1007/s41095-015-0004-6
# Copyright (c) 2023 Miller Cy Chan

class LSFilter:
    def __init__(self, pixels, side = 5, beta = 1500, lamda = 1e-4):
        self._height, self._width, _ = pixels.shape
        self._pixels = pixels
        self._side = np.clip(side, 3, 5)
        self._offset = self._side // 2
        self._beta = lamda
        self._lamda = beta / 250 if beta > 500 else beta / 50


    def getValue(self, x, y, x1, y1):
        if x1 < 0 or y1 < 0 or x1 >= self._width or y1 >= self._height:
            return 0
        if x == x1 and y == y1:
            return 1

        pixels = self._pixels
        color_diff = np.sum((pixels[y1, x1] - pixels[y, x]) ** 2)
        return -np.exp(-self._beta * color_diff)


    def getSmoothedKernel(self, y, x):
        offset, side = self._offset, self._side
        I = np.eye(side)
        laplacian = np.zeros(I.shape)
        for j in range(side):
            y1 = y + j - offset
            for i in range(side):
                x1 = x + i - offset
                laplacian[j, i] = self.getValue(x, y, x1, y1)

        result = I * self._lamda + laplacian @ laplacian.T
        return np.linalg.inv(result) * self._lamda


    def doFilter(self, y, x):
        offset, pixels, side = self._offset, self._pixels, self._side
        width, height = self._width, self._height
        laplacian = self.getSmoothedKernel(y, x)

        acc = np.zeros(pixels.shape[2])
        for j in range(side):
            y1 = y + j - offset
            if y1 < 0 or y1 >= height:
                continue

            for i in range(side):
                x1 = x + j - offset
                if x1 < 0 or x1 >= width:
                    continue

                acc += pixels[y1, x1] * laplacian[j, i]
        return np.clip(acc, 0, 255)


    def progress(self, percent_complete):
        if percent_complete < 100:
            print(int(percent_complete), "percent complete", end='\r')
        else:
            print("Well done!         ", end='\r')
            print("")


    def filter(self):
        pixels, qPixels = self._pixels, np.zeros(self._pixels.shape, np.uint8)
        width, height = self._width, self._height
        k = 0
        for y in range(height):
            for x in range(width):
                qPixels[y, x] = self.doFilter(y, x)
                k += 1
                self.progress(k * 100 / (width * height))
        return qPixels
