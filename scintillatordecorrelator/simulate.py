import numpy as np
from scipy.fft import fft2, ifft2
from scipy.integrate import dblquad


class Convolve:
    """Convolve the data in Fourier space using a PRF."""

    def __init__(self, prf):
        self.prf = prf
        self._otf = None
        self._otf_shape = None
        self._fact = None

    def __call__(self, im):
        if self._otf_shape is None or im.shape != self._otf_shape:
            self._otf = fft2(self.prf, im.shape)
            self._otf_shape = im.shape
        out = np.real(ifft2(fft2(im) * self._otf))
        out = np.roll(out, shift=-self.prf.shape[0] // 2 + 1, axis=0)
        out = np.roll(out, shift=-self.prf.shape[1] // 2 + 1, axis=1)
        return out


def double_laplace_pdf(x, y, b):
    return (1 / (2 * b)) * np.exp(- (np.abs(x) + np.abs(y)) / b)


def discretized_double_laplace(shape, b=0.5):
    width, height = shape
    image = np.zeros((height, width))

    # Pixel boundaries
    x_edges = np.linspace(-width / 2, width / 2, width + 1)
    y_edges = np.linspace(-height / 2, height / 2, height + 1)

    for i in range(width):
        for j in range(height):
            # Integrate over each pixel area
            x_start, x_end = x_edges[i], x_edges[i + 1]
            y_start, y_end = y_edges[j], y_edges[j + 1]
            pixel_integral, _ = dblquad(
                double_laplace_pdf,
                y_start, y_end,
                lambda x: x_start, lambda x: x_end, args=(b,))
            image[j, i] = pixel_integral

    return image


def simulate_data(
    img,
    prf,
    nr_ims=1000,
    sigma=0.0,
):
    conv = Convolve(prf)

    out = []
    for _ in range(nr_ims):
        img_noisy = np.copy(img)
        img_noisy = np.random.poisson(img_noisy)
        img_noisy = conv(img_noisy)
        img_noisy += np.random.normal(0, sigma, img_noisy.shape)
        out.append(img_noisy)
    return np.asarray(out)
