import numpy as np
import torch


PI = np.pi
TWO_PI = 2. * np.pi
INV_PI = 1. / np.pi
SQRT_TWO = np.sqrt(2.)
INV_TWOPI = 1. / 2. / np.pi  # 1 / 2 / pi
INV_FOURPI = 1. / 4. / np.pi  # 1 / 4 / pi
SQRT_PI = np.sqrt(np.pi)
INV_SQRT_TWO = 1. / np.sqrt(2.)
PI_OVER_4 = np.pi / 4.  # pi / 4
PI_OVER_2 = np.pi / 2.  # pi / 2

########################################################################################################################
# functions for np.ndarray


def magnitude(a, axis=-1):
    return np.sqrt(np.sum(a * a, axis=axis))


def normalize(a, axis=-1):
    return a / np.expand_dims(magnitude(a, axis), axis)


def dot(a, b, axis=-1):
    return np.sum(a * b, axis=axis)


def dot_abs(a, b, axis=-1):
    return np.abs(dot(a, b, axis))


def lerp(a, b, r):
    return a * (1.0 - r) + b * r


def saturate(a, b):
    return np.clip(a + b, 0., 1.)


########################################################################################################################
# functions for torch.Tensor


def magnitude_t(a, dim=-1):
    return torch.sqrt(torch.sum(a * a, dim=dim))


def normalize_t(a, dim=-1):
    return a / torch.unsqueeze(magnitude_t(a, dim), dim)


def dot_t(a, b, dim=-1):
    return torch.sum(a * b, dim=dim)


def dot_t_abs(a, b, dim=-1):
    return torch.abs(dot_t(a, b, dim))


def lerp_t(a, b, r):
    return a * (1.0 - r) + b * r


def saturate_t(a, b):
    return torch.clip(a + b, 0., 1.)
