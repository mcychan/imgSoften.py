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
        self._width, self._height, self._channel = pixels.shape
        self._offset = 1
        self._pixels = np.copy(pixels)

    def flux_derivative(self, news):
        kappa = self._kappa
        if self._useExp:
            return np.exp(-1 * ((news / kappa) ** 2))

        return 1 / (1 + ((news / kappa) ** 2))


    def getValue(self, center, cNews, news):
        return center + self._lamda * np.sum(cNews * news, axis = 0)


    def doFilter(self, x, y):
        offset, pixels = self._offset, self._pixels
        width, height = self._width, self._height
        north, east, west, south = pixels[x - offset, y], pixels[x, y + offset], pixels[x, y - offset], pixels[x + offset, y]
        center = pixels[x, y]

        news = np.array([north - center, east - center, west - center, south - center])
        cNews = self.flux_derivative(news)
        return self.getValue(center, cNews, news)


    def filter(self):
        pixels = self._pixels
        iteration, offset, width, height = self._iteration, self._offset, self._width, self._height
        for z in range(iteration):
            for y in range(offset, height - offset):
                for x in range(offset, width - offset):
                    pixels[x, y] = self.doFilter(x, y)
        return pixels
