import numpy as np

# A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images.
# https://en.wikipedia.org/wiki/Bilateral_filter
# Copyright (c) 2023 Miller Cy Chan

class BilateralFilter:
    def __init__(self, pixels, diameter = 5, sigmaI = 12.0, sigmaS = 16.0):
        self._height, self._width, _ = pixels.shape
        self._pixels = pixels
        self._diameter = diameter
        self._sigmaI = sigmaI
        self._sigmaS = sigmaS


    @staticmethod
    def distance(x, y, i, j):
        return np.sqrt((x - i) ** 2 + (y - j) ** 2)


    @staticmethod
    def gaussian(diff, sigma):
        sigma2 = sigma ** 2
        return np.exp(-(diff ** 2) / (2 * sigma2) / (2 * np.pi * sigma2))


    def doFilter(self, y, x):
        distance, gaussian = BilateralFilter.distance, BilateralFilter.gaussian
        pixels, diameter = self._pixels, self._diameter
        width, height = self._width, self._height
        sigmaI, sigmaS = self._sigmaI, self._sigmaS

        iFiltered, wP = np.zeros(pixels.shape[2]), np.zeros(pixels.shape[2])
        x1, y1, half = 0, 0, diameter // 2

        for j in range(diameter):
            y1 = y - half + j
            if y1 < 0 or y1 >= height:
                continue

            for i in range(diameter):
                x1 = x - half + i
                if x1 < 0 or x1 >= width:
                    continue

                gs = gaussian(distance(x, y, x1, y1), sigmaS)
                w = gs * gaussian(pixels[y1, x1] - pixels[y, x], sigmaI)
                iFiltered += pixels[y1, x1] * w
                wP += w
        return np.rint(iFiltered / wP)


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
