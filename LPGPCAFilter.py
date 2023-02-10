import numpy as np

# Lei Zhang, Weisheng Dong, David Zhang, Guangming Shi
# Two-stage Image Denoising by Principal Component Analysis with Local
# Pixel Grouping, Pattern Recognition, vol. 43, issue 4, pp. 1531-1549, Apr. 2010
# Copyright (c) 2023 Miller Cy Chan

class LPGPCAFilter:
    def __init__(self, pixels, v = 20, S = 20, t = 3, nblk = 250):
        self._width, self._height, self._channels = pixels.shape
        self._pixels = pixels.astype(dtype = np.float32)

        noise = np.random.rand(self._width, self._height, self._channels) ** 2
        noise /= np.sqrt(np.sum(noise) / np.size(noise))
        self._noisePixels = np.clip(self._pixels + v * noise, 0, 255).astype(dtype = np.uint8)
        self._nblk, self._S, self._t, self._v = nblk, S, t, v


    def getPca(self, X):
        M, N = X.shape[:2]
        mx = np.array([np.mean(X, axis = 1)] * N).T
        X1 = X - mx
        CovX = X1 @ X1.T / (N - 1)
        V, P = np.linalg.eigh(CovX)
        ind = np.argsort(-V)
        V = V[ind]
        P = P[:, ind].T
        Y = P @ X1
        return [Y, P, V, mx]


    def LPG_new(self, X, row, col, off, nv, S, I):
        N, M = I.shape[:2]
        f2 = X.shape[1]
        rmin, rmax = max(row - S, 0), min(row + S, N)
        cmin, cmax = max(col - S, 0), min(col + S, M)
        idx = I[rmin: rmax, cmin: cmax].reshape(-1)
        B, v = X[idx, :], X[off, :]

        dis = np.mean((B - v[: f2]) ** 2, axis = -1)
        ind = np.argsort(dis)
        return idx[ind[: nv]]


    def dim_reduction(self, X):
        n = int(np.floor(X.shape[0] * 0.4))
        P = self.getPca(X)[1]
        return P[: n, :] @ X


    def doFilter(self, nI, v2):
        k, nblk, s, S = 0, self._nblk, 2, self._S
        ch, w, h = self._channels, self._width, self._height
        b = 2 * self._t + 1
        b2 = b ** 2
        N, M = w - b + 1, h - b + 1
        L = N * M
        c = np.arange(M)[::s]
        c = np.append(c, c[-1] + 1)
        r = np.arange(N)[::s]
        r = np.append(r, r[-1] + 1)
        X = np.zeros((b2 * ch, L))

        for i in range(b):
            x = N + i
            for j in range(b):
                channels = [k, k + b2, k + b2 * 2][: ch]
                y = M + j

                for l in range(ch):
                    X[channels[l], :] = nI[j : y, i : x, l].T.reshape(-1)
                k += 1

        XT = X.T
        # XT = self.dim_reduction(X).T
        I = np.arange(L).reshape(M, N).T
        N1, M1 = len(r), len(c)
        Y = np.zeros((b2 * ch, N1 * M1))

        for i in range(N1):
            row = r[i]
            for j in range(M1):
                col = c[j]
                off, off1 = col * N + row, j * N1 + i

                indc = self.LPG_new(XT, row, col, off, nblk, S, I)
                coe, P, V, mX = self.getPca(X[:, indc])
                py = np.mean(coe ** 2, axis = 1).astype(np.float32)
                px = py - v2
                px[px < 0], py[py == 0] = 0, np.finfo(float).eps
                wei = px / py
                Y[:, off1] = P.T @ (coe[:, 0] * wei) + mX[:, 0]

        # Output the processed image
        dI, im_wei = np.zeros((w, h, ch)), np.zeros((w, h, ch))
        k = 0
        for i in range(b):
            ri = r + i
            for j in range(b):
                channels, cj = [k, k + b2, k + b2 * 2][: ch], c + j

                for l in range(ch):
                    layer = Y[channels[l], :].T.reshape(N1, M1)
                    for n in range(N1):
                        dI[ri[n], cj, l] += layer[n]
                        im_wei[ri[n], cj, l] += 1
                k += 1

        dI /= im_wei + np.finfo(float).eps
        return np.clip(dI, 0, 255)


    def stage1(self, nI):
        v2 = self._v ** 2
        return self.doFilter(nI, v2)


    def stage2(self, nI, v1):
        v2 = (v1 * 0.37) ** 2
        return self.doFilter(nI, v2)


    def filter(self):
        width, height, channels = self._width, self._height, self._channels

        nI = self._noisePixels
        qPixels = self.stage1(nI)
        diff, v2 = qPixels - nI, self._v ** 2
        vd = np.mean(v2 + np.mean(diff ** 2, axis = (0, 1)))
        v1 = np.sqrt(abs(vd))
        qPixels = self.stage2(qPixels, v1)
        return qPixels.astype(dtype = np.uint8)
