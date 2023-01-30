import numpy as np

# Fast Global Image Smoothing based on Weighted Least Squares
# D. Min, S. Choi, J. Lu, B. Ham, K. Sohn, and M. N. Do,
# IEEE Trans. Image Processing, vol. no. pp., 2014.
# Copyright (c) 2023 Miller Cy Chan

class FGSFilter:
    def __init__(self, pixels, sigma = 0.1, lamda = 900, joint_pixels = None, num_iterations = 3, attenuation = 4):
        self._width, self._height, channels = pixels.shape
        self._pixels = np.copy(pixels)
        self._pixels = np.array(self._pixels.reshape(self._width * self._height, channels), dtype = np.float32)
        if joint_pixels is not None:
            self._width, self._height, channels = joint_pixels.shape
        self._joint_pixels = np.copy(self._pixels) if joint_pixels is None else joint_pixels.reshape(self._width * self._height, channels)
        self._lamda_in = 1.5 * lamda * np.power(4.0, num_iterations - 1) / (np.power(4.0, num_iterations) - 1)
        self._num_iterations = num_iterations
        self._attenuation = attenuation
        self._BLFKernelI = np.exp(-np.sqrt(np.arange((255 ** 2) * 3 + 1) / sigma / 255))


    def doFilter(self, n, length, horizontal):
        a_vec, b_vec, c_vec = np.zeros(length), np.zeros(length), np.zeros(length)
        joint_pixels, pixels, width, height = self._joint_pixels, self._pixels, self._width, self._height
        BLFKernelI, idx0 = self._BLFKernelI, n * width if horizontal else n
        c0, lamda_in = joint_pixels[idx0], self._lamda_in
        aIdx, bIdx, cIdx = 1, 0, 0
        idx0 += 1

        for _ in range(1, length):
            idx0 += 0 if horizontal else width - 1
            c1 = joint_pixels[idx0]
            idx0 += 1
            color_diff = int(np.sum((c0 - c1) ** 2))
            c_vec[cIdx] = a_vec[aIdx] = -lamda_in * BLFKernelI[color_diff]
            aIdx += 1
            cIdx += 1
            c0 = c1

        b_vec = 1 - a_vec - c_vec

        # solver
        c_vec[0] /= b_vec[0]
        idx0 = n * width if horizontal else n
        pixels[idx0] /= b_vec[0]
        aIdx, bIdx, cIdx = 1, 1, 0

        c0 = pixels[idx0]
        idx0 += 1
        idx1 = 1 + n * width if horizontal else n + width

        for _ in range(1, length):
            m = 1.0 / (b_vec[bIdx] - a_vec[aIdx] * c_vec[cIdx])
            cIdx += 1
            c_vec[cIdx] *= m

            idx0 += 0 if horizontal else width - 1
            c1 = pixels[idx0]
            idx0 += 1
            c0 = (c1 - a_vec[aIdx] * c0) * m

            bIdx += 1
            aIdx += 1
            pixels[idx1] = c0
            idx1 += 1 if horizontal else width

        cIdx = length - 2
        idx0 -= 1
        c0 = pixels[idx0]
        idx1 -= 1 if horizontal else 2 * width - 1

        for _ in range(length - 1, 0, -1):
            idx0 -= 1 if horizontal else width
            c1 = pixels[idx0]
            m = c_vec[cIdx]
            cIdx -= 1
            c0 = c1 - m * c0
            idx1 -= 1
            pixels[idx1] = c0
            idx1 -= 0 if horizontal else width - 1


    def filter(self):
        num_iterations, width, height = self._num_iterations, self._width, self._height
        pixels = self._pixels
        for z in range(num_iterations):
            for y in range(height):
                self.doFilter(y, width, True)

            for x in range(width):
                self.doFilter(x, height, False)

            self._lamda_in /= self._attenuation

        return pixels.reshape(width, height, pixels.shape[1]).astype(dtype = np.uint8)
