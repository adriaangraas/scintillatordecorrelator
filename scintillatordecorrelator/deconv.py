import numpy as np
from scipy.fft import fft2, ifft2

class Deconvolve:
    """Deconvolve the data in Fourier space using a point-response function."""

    def __init__(self, prf, reg_constant=0.0):
        self._reg = reg_constant
        self.prf = prf
        self._otf = None
        self._otf_shape = None
        self._fact = None

    def __call__(self, im):
        if self._otf_shape is None or im.shape != self._otf_shape:
            self._otf = fft2(self.prf, im.shape)
            self._otf_shape = im.shape
            self._fact = np.conj(self._otf) / (np.abs(self._otf) ** 2 + self._reg)

        out = np.real(ifft2(fft2(im) * self._fact))
        out = np.roll(out, shift=self.prf.shape[0] // 2, axis=0)
        out = np.roll(out, shift=self.prf.shape[1] // 2, axis=1)
        return out
