import cv2
import numpy as np
from BilateralFilter import BilateralFilter

# Wada, Naofumi & Kazui, Masato & Haseyama, Miki. (2015).
# Extended Joint Bilateral Filter for the Reduction of Color Bleeding in Compressed Image and Video.
# ITE Transactions on Media Technology and Applications. 3. 95-106. 10.3169/mta.3.95. 
# Copyright (c) 2023 Miller Cy Chan

class EJBilateralFilter:
    def __init__(self, pixels, diameter = 5, sigmaS = 4.0):
        self._height, self._width, _ = pixels.shape
        self._pixels = np.array(cv2.cvtColor(pixels, cv2.COLOR_BGR2YCR_CB), dtype = np.float32)
        self._diameter = diameter
        pixelY, pixelCr, pixelCb = self._pixels[:, :, 0], self._pixels[:, :, 1], self._pixels[:, :, 2]
        self._sigmaS, self._sigmaR = sigmaS, np.array([np.std(pixelY), np.std(pixelCr), np.std(pixelCb)])


    def doFilter(self, y, x):
        distance, gaussian = BilateralFilter.distance, BilateralFilter.gaussian
        pixels, diameter = self._pixels, self._diameter
        width, height = self._width, self._height
        sigmaS, sigmaR = self._sigmaS, self._sigmaR

        iFiltered, wP = np.zeros(pixels.shape[2]), np.zeros(pixels.shape[2])
        x1, y1, half = 0, 0, diameter // 2

        for i in range(diameter):
            y1 = y - half + i
            if y1 < 0 or y1 >= height:
                continue

            for j in range(diameter):
                x1 = x - half + j
                if x1 < 0 or x1 >= width:
                    continue

                gs = gaussian(distance(x, y, x1, y1), sigmaS)
                w = gs * np.prod(gaussian(pixels[y1, x1] - pixels[y, x], sigmaR), axis=0)
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
        return cv2.cvtColor(qPixels, cv2.COLOR_YCR_CB2BGR)
