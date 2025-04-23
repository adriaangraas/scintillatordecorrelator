import numpy as np
from scipy.linalg import sqrtm
from scipy.signal import correlate

def estimate_autocorrelation(
    data,
    estimator='differences',
    out_shape=(5, 5),
    dt=3
):
    diff = estimator == 'differences'
    if not diff:  # use the mean
        V = np.sqrt(np.mean(data, axis=0))
        WD = data - V
        WD = np.divide(WD, V, where=V != 0)
    else:
        assert data.shape[0] > dt, f"Not enough images for dt={dt}"
        WD = data[:-dt] - data[dt:]
        assert WD.shape[0] >= 1
        pairwise_var = np.var(data[:-dt] - data[dt:], axis=0)
        np.clip(pairwise_var, 0.0, None, out=pairwise_var)
        pairwise_std = np.sqrt(pairwise_var)
        WD = np.divide(WD, pairwise_std, where=pairwise_std != 0)

    out = np.zeros(out_shape, dtype=np.float64)
    padd = np.max(out_shape) + 1
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            i_ = i - out.shape[0] // 2
            j_ = j - out.shape[1] // 2
            data1 = np.roll(WD, axis=1, shift=i_)
            data2 = np.roll(WD, axis=2, shift=j_)
            data1 = data1[:, padd:-padd, padd:-padd]
            data2 = data2[:, padd:-padd, padd:-padd]
            out[i, j] = np.mean(data1 * data2)

    return out


def convolution_matrix(kernel, image_shape):
    height, width = image_shape
    operator = np.zeros((height * width, height * width))
    for i in range(height):
        for j in range(width):
            basis_image = np.zeros(image_shape)
            basis_image[i, j] = 1
            convolved_image = correlate(basis_image, kernel, mode='same')
            operator[:, i * width + j] = convolved_image.flatten()
    return operator


def solve_kernel(h_conv_h):
    out_shape = list((s // 2 + 1 for s in h_conv_h.shape))
    op = convolution_matrix(h_conv_h, out_shape)
    H = sqrtm(op)
    h = np.real(H[op.shape[0] // 2]).reshape(out_shape)
    return h