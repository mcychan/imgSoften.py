import numpy as np

# A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images.
# https://en.wikipedia.org/wiki/Bilateral_filter
# Copyright (c) 2023 Miller Cy Chan

class BilateralFilter:
    def __init__(self, pixels, diameter = 5, sigmaI = 12.0, sigmaS = 16.0):
        self._width, self._height, _ = pixels.shape
        self._pixels = pixels
        self._diameter = diameter
        self._sigmaI = sigmaI
        self._sigmaS = sigmaS


    @staticmethod
    def distance(x, y, i, j):
        return np.sqrt((x - i) ** 2 + (y - j) ** 2)


    @staticmethod
    def gaussian(pixel, sigma):
        return np.exp(-(pixel ** 2)/(2 * (sigma ** 2)) / (2 * np.pi * (sigma ** 2)))


    def doFilter(self, x, y):
        pixels, diameter = self._pixels, self._diameter
        width, height = self._width, self._height
        sigmaI, sigmaS = self._sigmaI, self._sigmaS

        iFiltered, wP = np.zeros(pixels.shape[2]), np.zeros(pixels.shape[2])
        x1, y1, half = 0, 0, diameter // 2

        for i in range(diameter):
            x1 = x - half + i
            if x1 < 0 or x1 >= width:
                continue

            for j in range(diameter):
                y1 = y - half + j
                if y1 < 0 or y1 >= height:
                    continue

                gs = BilateralFilter.gaussian(BilateralFilter.distance(x, y, x1, y1), sigmaS)
                w = gs * BilateralFilter.gaussian(pixels[x1, y1] - pixels[x, y], sigmaI)
                iFiltered += pixels[x1, y1] * w
                wP += w
        return np.rint(iFiltered / wP)


    def filter(self):
        pixels, qPixels = self._pixels, np.zeros(self._pixels.shape, np.uint8)
        width, height = self._width, self._height
        for y in range(height):
            for x in range(width):
                qPixels[x, y] = self.doFilter(x, y)
        return qPixels
