import numpy as np

# Anisotropic diffusion, also called Perona–Malik diffusion, is a technique aiming at reducing image noise without removing significant parts of the image content,
# typically edges, lines or other details that are important for the interpretation of the image.
# Copyright (c) 2023 Miller Cy Chan

class AnisotropicDiffusionFilter:
    def __init__(self, pixels, iteration = 20, lamda = 0.5, kappa = 3.0, useExp = False):
        self._iteration = iteration
        self._lamda = lamda
        self._kappa = kappa
        self._useExp = useExp
        self._height, self._width, self._channel = pixels.shape
        self._offset = 1
        self._pixels = np.copy(pixels)

    def flux_derivative(self, news):
        kappa = self._kappa
        if self._useExp:
            return np.exp(-(((news / kappa) ** 2)))

        return 1 / (1 + ((news / kappa) ** 2))


    def getValue(self, center, cNews, news):
        return center + self._lamda * np.sum(cNews * news, axis = 0)


    def doFilter(self, y, x):
        offset, pixels = self._offset, self._pixels
        width, height = self._width, self._height
        north, east, west, south = pixels[y, x - offset], pixels[y + offset, x], pixels[y - offset, x], pixels[y, x + offset]
        center = pixels[y, x]

        news = np.array([north, east, west, south]) - center
        cNews = self.flux_derivative(news)
        return self.getValue(center, cNews, news)


    def progress(self, percent_complete):
        if percent_complete < 100:
            print(int(percent_complete), "percent complete", end='\r')
        else:
            print("Well done!         ", end='\r')
            print("")


    def filter(self):
        qPixels = self._pixels
        num_iterations, offset, width, height = self._iteration, self._offset, self._width, self._height
        for z in range(num_iterations):
            self.progress(z * 100 / num_iterations)
            for y in range(offset, height - offset):
                for x in range(offset, width - offset):
                    qPixels[y, x] = self.doFilter(y, x)
        self.progress(100)
        return qPixels
