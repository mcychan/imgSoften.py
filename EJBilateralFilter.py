import cv2
import numpy as np

# Wada, Naofumi & Kazui, Masato & Haseyama, Miki. (2015).
# Extended Joint Bilateral Filter for the Reduction of Color Bleeding in Compressed Image and Video.
# ITE Transactions on Media Technology and Applications. 3. 95-106. 10.3169/mta.3.95. 
# Copyright (c) 2023 Miller Cy Chan

def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(pixel, sigma):
    return np.exp(-(pixel ** 2)/(2 * (sigma ** 2)) / (2 * np.pi * (sigma ** 2)))


class EJBilateralFilter:
    def __init__(self, pixels, diameter = 5, sigmaS = 4.0):
        self._width, self._height, _ = pixels.shape
        self._pixels = np.array(cv2.cvtColor(pixels, cv2.COLOR_BGR2YCR_CB), dtype = np.float32)
        self._diameter = diameter
        pixelY, pixelCr, pixelCb = self._pixels[:, :, 0], self._pixels[:, :, 1], self._pixels[:, :, 2]
        self._sigmaS, self._sigmaR = sigmaS, np.array([np.std(pixelY), np.std(pixelCr), np.std(pixelCb)])


    def doFilter(self, x, y):
        pixels, diameter = self._pixels, self._diameter
        width, height = self._width, self._height
        sigmaS, sigmaR = self._sigmaS, self._sigmaR

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

                gs = gaussian(distance(x, y, x1, y1), sigmaS)
                w = gs * np.prod(gaussian(pixels[x1, y1] - pixels[x, y], sigmaR), axis=0)
                iFiltered += pixels[x1, y1] * w
                wP += w
        return np.rint(iFiltered / wP)


    def filter(self):
        pixels, qPixels = self._pixels, np.zeros(self._pixels.shape, np.float32)
        width, height = self._width, self._height
        for y in range(height):
            for x in range(width):
                qPixels[x, y] = self.doFilter(x, y)
        return cv2.cvtColor(np.array(qPixels, dtype = np.uint8), cv2.COLOR_YCR_CB2BGR)
