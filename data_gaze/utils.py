import torch

from nerg.debug import tinfo


# this needs to run at full precision otherwise NaNs for sigma<0.01
@torch.cuda.amp.autocast(dtype=torch.float32)
def unit_sphere_smoothing(rays, values, sigma=0.01):
    """
    Smoothes predictions for nearby rays on the sphere.
    Args:
        values: N values to smooth
        rays: N,3 rays on unit sphere associated with values
        sigma: smaller values lead to less smoothing
    Returns:
    """

    # Pairwise distances on the unit sphere
    tinfo('rays', rays)
    cosine_sim = torch.matmul(rays, rays.T)  # Cosine similarity between rays
    # tinfo('cosine_sim', cosine_sim)

    distances = (1 - cosine_sim)  # distances are in range [0,2]
    # tinfo('distances', distances)

    sigma2 = 2 * sigma ** 2
    # print('sigma2', sigma2)
    # Apply Gaussian kernel
    weights = torch.exp(-(distances * distances) / sigma2)
    weights_sum = weights.sum(dim=1)  # apply normalization after summing

    # tinfo('weights', weights)
    # tinfo('weights_sum', weights_sum)

    # Compute smoothed values by applying weighted sum
    smoothed_values = torch.matmul(weights, values)
    # tinfo('smoothed_values pre', smoothed_values)
    smoothed_values /= weights_sum
    # tinfo('smoothed_values post', smoothed_values)

    return smoothed_values
