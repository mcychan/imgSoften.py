import numpy as np

# Wang, H., Cao, J., Liu, X. et al. Least-squares images for edge-preserving smoothing.
# Comp. Visual Media 1, 27–35 (2015).
# https://doi.org/10.1007/s41095-015-0004-6
# Copyright (c) 2023 Miller Cy Chan

class LSFilter:
    def __init__(self, pixels, side = 3, beta = 1500, lamda = 1e-4):
        self._height, self._width, _ = pixels.shape
        self._pixels = pixels
        self._side = np.clip(side, 3, 5)
        self._offset, self._I = self._side // 2, np.eye(side)
        self._beta = lamda # swap lamda and beta due to exp(-1500 * (value > 1)) tends to 0
        self._lamda = beta


    def getValue(self, y, x, y1, x1):
        if x1 < 0 or y1 < 0 or x1 >= self._width or y1 >= self._height:
            return (0, 0)
        if x == x1 and y == y1:
            return (1, 1)

        pixels = self._pixels
        diff = np.abs(pixels[y1, x1] - pixels[y, x]).astype(dtype = np.float32) + 1
        return (np.exp(-self._beta * diff.T.dot(diff)), -1)


    def getSmoothedKernel(self, y, x):
        offset, side, I = self._offset, self._side, self._I
        filter, L = np.zeros(I.shape).astype(bool), np.zeros(I.shape).astype(dtype = np.float32)
        for j in range(side):
            y1 = y + j - offset
            for i in range(side):
                x1 = x + i - offset
                tuple = self.getValue(y, x, y1, x1)
                L[j, i] = tuple[0]
                if tuple[1] < 0:
                    filter[j, i] = True

        L = self._lamda * I + L.T.dot(L)
        L = self._lamda * np.linalg.pinv(L)
        divisor = np.sum(L[filter])
        if divisor != 0:
            L[filter] /= -divisor
        return L


    def doFilter(self, y, x):
        offset, pixels, side = self._offset, self._pixels, self._side
        width, height = self._width, self._height
        kernel = self.getSmoothedKernel(y, x)

        acc = pixels[y, x].astype(dtype = np.float32)
        for j in range(side):
            y1 = y + j - offset
            if y1 < 0 or y1 >= height:
                continue

            for i in range(side):
                x1 = x + i - offset
                if x1 < 0 or x1 >= width:
                    continue

                acc += pixels[y1, x1] * kernel[j, i]
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
