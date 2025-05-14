import numpy as np
from scipy.linalg import sqrtm
from scipy.signal import correlate

def estimate_autocorrelation(
    data,
    estimator='mean',
    out_shape=(5, 5),
    dt=3
):
    assert data.ndim == 3

    if estimator == 'mean':
        mean = np.mean(data, axis=0)
        noise = data - mean  # noise component, which is anisotropic
        sigma_squared = mean  # mean equals the variance for Poisson
        # stabilization:
        noise = np.divide(noise, np.sqrt(sigma_squared), where=sigma_squared != 0)
    elif estimator == 'differences':
        assert data.shape[0] > dt, f"Not enough images for dt={dt}"
        # noise via differences doubles the variance
        noise = data[:-dt] - data[dt:]
        sigma_squared = np.var(noise, axis=0)
        assert noise.shape[0] >= 1
        noise = np.divide(noise, np.sqrt(sigma_squared), where=sigma_squared != 0)
    else:
        raise ValueError(f"Unknown estimator '{estimator}'.")

    out = np.zeros(out_shape, dtype=np.float64)
    padd = np.max(out_shape) + 1
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            i_ = i - out.shape[0] // 2
            j_ = j - out.shape[1] // 2
            noise_rolled = np.roll(noise, axis=1, shift=i_)
            noise_rolled = np.roll(noise_rolled, axis=2, shift=j_)
            covariance_image = (
                noise[:, padd:-padd, padd:-padd]
                * noise_rolled[:, padd:-padd, padd:-padd])
            out[i, j] = np.mean(covariance_image)

    return out


def correlation_map(
    data,
    estimator='mean',
    dist=(7, 7),
    dt=3
):
    if estimator == 'mean':
        mean = np.mean(data, axis=0)
        noise = data - mean
        sigma_squared = mean  # mean equals the variance for Poisson
        noise = np.divide(noise, np.sqrt(sigma_squared), where=sigma_squared != 0)
    elif estimator == 'differences':
        noise = data[:-dt] - data[dt:]
        pairwise_var = np.var(data[:-dt] - data[dt:], axis=0)
        np.clip(pairwise_var, 0.0, None, out=pairwise_var)
        pairwise_std = np.sqrt(pairwise_var)
        noise = np.divide(noise, pairwise_std, where=pairwise_std!=0)
    else:
        raise ValueError(f"Unknown estimator '{estimator}'.")

    out = np.zeros(data.shape[1:])
    for i in range(dist[0]):
        for j in range(dist[1]):
            i_ = i - dist[0] // 2
            j_ = j - dist[1] // 2
            if i_ == j_ == 0:
                continue
            noise_rolled = np.roll(noise, axis=1, shift=i_)
            noise_rolled = np.roll(noise_rolled, axis=2, shift=j_)
            out += np.mean(noise * noise_rolled, axis=0)

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