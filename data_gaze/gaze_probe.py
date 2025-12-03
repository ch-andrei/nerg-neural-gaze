import os

import cartopy
import numpy as np
import shapely
import torch
from matplotlib import pyplot as plt

from scipy.special import logsumexp
from scipy.interpolate import griddata

from common.logging import log
from common.numpy_utils import TemporaryNumpySeed
from common.vector_math import normalize, magnitude
from data_gaze.utils import unit_sphere_smoothing
from spherical_kde.spherical_kde.distributions import SphericalKDE, sample_uniform_sphere, make_fibonacci_unit_sphere
from spherical_kde.spherical_kde.utils import polar_from_decra, cartesian_from_polar, polar_from_cartesian, \
    decra_from_polar


class GazeProbe(object):
    def __init__(self,
                 ro_near,  # gaze ray origin positions (3,N)
                 rd_near,  # gaze ray normalized directions (3,N)
                 dists,  # distance between gaze ray origin positions and the gaze query point
                 dist_weighted=False,
                 preload_num_gaze_rays=-1
                 ):
        self.ro = np.mean(ro_near, axis=1).reshape(3, 1)  # average ro_near over N then reshape to (3,1)
        self.dist_mean = np.mean(dists)  # average dist
        self.dist_max = np.max(dists)  # maximum dist
        self.num_rays = rd_near.shape[1]

        self.density = SphericalKDE(
            normalize(rd_near, axis=0),  # note: KDE uses shape (3, N), make sure that vectors are normalized
            weights=(1.0 - dists / dists.max()) if dist_weighted else None  # weight samples by distance to query point
        )

        self.preload_gaze_rays = 0 < preload_num_gaze_rays
        if self.preload_gaze_rays:
            self.rays, self.log_prob = self.sample_gaze_rays_uniform(preload_num_gaze_rays)

    def sample_gaze_rays(self, num_rays, allow_preloaded=True):
        if self.preload_gaze_rays and allow_preloaded:
            return self.rays, self.log_prob  # return precomputed samples
        else:
            return self.sample_gaze_rays_uniform(num_rays)  # new sample

    def sample_gaze_rays_uniform(self, num_rays):
        rays = sample_uniform_sphere(num_rays)
        log_prob = self.density.evaluate_log_prob_density(rays)
        return rays, log_prob


def make_figure_gaze_probe(dpi=300, use_square_projection=False):
    plt.figure(dpi=dpi)
    if use_square_projection:
        ax = plt.subplot(111, projection=cartopy.crs.PlateCarree())
    else:
        ax = plt.subplot(111, projection=cartopy.crs.Mollweide())
    return ax


def finish_figure_gaze_probe(ax, legend=False):
    ax.gridlines(alpha=0.5)
    ax.coastlines(linewidth=0.1)
    if legend:
        plt.legend()
    ax.axis('equal')
    plt.colorbar()
    plt.draw()


class GazeProbePlotHelper(object):
    def __init__(self, save_path=None, dpi=300, legend=False, use_square_projection=False):
        self.save_path = save_path
        self.legend = legend
        self.ax = make_figure_gaze_probe(dpi, use_square_projection=use_square_projection)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        finish_figure_gaze_probe(self.ax, self.legend)
        if self.save_path is not None:
            log(f"Saving figure to {self.save_path}")
            # Note: if not manually removed, file's last_modified does not change
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            plt.savefig(self.save_path)
            plt.close()  # close figure after saving


def plot_gaze_probe_kde(gaze_probe: GazeProbe):
    """ Plot the KDE on an axis.
    """
    # Compute the kernel density estimate on an equiangular grid
    ra = np.linspace(-180, 180, 360)
    dec = np.linspace(-89, 89, 180)
    X, Y = np.meshgrid(ra, dec)

    phi, theta = polar_from_decra(X, Y)
    x = cartesian_from_polar(phi, theta)
    P = np.exp(gaze_probe.density.evaluate_log_prob_density(x))

    # from pysaliency.plotting import visualize_distribution
    # visualize_distribution(P)
    # plt.show()
    # exit()
    # Plot the countours on a suitable equiangular projection
    plt.contourf(X, Y, P, levels=25, transform=cartopy.crs.PlateCarree())
    plt.scatter(X, Y, c=P, cmap='viridis', transform=cartopy.crs.PlateCarree())


def plot_gaze_rays_p(x, P):
    """ Plot the KDE on an axis. x must be of shape (3,...)
    """

    ra = np.linspace(-180, 180, 360)
    dec = np.linspace(-89, 89, 180)
    X, Y = np.meshgrid(ra, dec)  # equally spaced grid on which to evaluate/display the density

    phi, theta = polar_from_cartesian(x)
    ra, dec = decra_from_polar(phi, theta)
    # plt.scatter(ra, dec, c=P, s=2, cmap='viridis', transform=cartopy.crs.PlateCarree())
    grid_z2 = griddata((ra, dec), P, (X, Y), method='linear')  # interpolate from available data to the grid

    plt.contourf(X, Y, grid_z2, levels=25, transform=cartopy.crs.PlateCarree())


def plot_gaze_rays_phi_theta(phi, theta, color, label=None):
    ra, dec = decra_from_polar(phi, theta)
    plt.plot(ra, dec, f'{color}.', ms=0.25, transform=cartopy.crs.PlateCarree(), label=label)


def plot_gaze_rays(x, color, label=None):
    phi, theta = polar_from_cartesian(x)
    plot_gaze_rays_phi_theta(phi, theta, color=color, label=label)


def plot_gaze_probe_data(gaze_probe: GazeProbe):
    print("Plotting gaze_probe:", gaze_probe.ro.flatten(), gaze_probe.num_rays)
    with GazeProbePlotHelper(legend=True):
        plt.title(f"GazeProbe pdf [KDE with {gaze_probe.num_rays} gaze rays]")
        plot_gaze_probe_kde(gaze_probe)  # prob map
        plot_gaze_rays(gaze_probe.density.mu, color='r', label='centers')
        plot_gaze_rays(gaze_probe.sample_gaze_rays_uniform(1000)[0],
                       color='k', label='random')


if __name__ == "__main__":
    # tests

    print("Gaze probe debugging...")

    def integrate_prob(gaze_probe):
        # integrate probabilities over a sphere
        title = 'Monte-carlo integration over random samples'
        Ns = (np.arange(100, step=5) + 1) * 100
        integrals = []
        for n in Ns:
            def sample_gaze_rays_fibonacci(num_rays):
                x = make_fibonacci_unit_sphere(num_rays)
                log_prob = gaze_probe.density.evaluate_log_prob_density(x)
                return x, log_prob

            x, log_prob = sample_gaze_rays_fibonacci(n)
            prob = np.exp(log_prob)
            # x, prob = gaze_sphere.sample_gaze_rays_uniform(n)
            monte_carlo_integral = (4 * np.pi * np.mean(prob))
            integrals.append(monte_carlo_integral)
            print(title, monte_carlo_integral)
        plt.figure(dpi=300)
        plt.title(title)
        plt.plot(Ns, integrals)
        plt.draw()

    with TemporaryNumpySeed():
        n_rd = 256
        rays_d = normalize(np.random.rand(n_rd, 3))
        # print('magnitude(rays_d)', magnitude(rays_d))

        gaze_probe = GazeProbe(
            np.array([0, 1, 0]).reshape(3, 1),
            rays_d.transpose(),
            np.zeros(1)
        )

        # integrate_prob(gaze_sphere)

        rays, log_prob = gaze_probe.sample_gaze_rays_uniform(500)
        prob = np.exp(log_prob)

        print('rays.shape', rays.shape)
        print('prob.shape', prob.shape, prob.min(), prob.max())
        print('gaze_sphere.ro', gaze_probe.ro)

        prob_t = torch.from_numpy(prob)
        rays_t = torch.from_numpy(rays.transpose())  # rays are 3,N - transpose for N,3
        prob_smooth = unit_sphere_smoothing(rays_t, prob_t, sigma=0.1).numpy()

        save_path = "results/test_att/results"
        name = "plot_test"
        i = 0
        # with GazeProbePlotHelper(os.path.join(save_path, f'{name}_{i:04d}_gaze_gt.png')):
        with GazeProbePlotHelper():
            plt.title(f"GazeProbe pdf [KDE with {gaze_probe.num_rays} gaze rays]")
            # ax = plt.subplot(111, projection=cartopy.crs.PlateCarree())
            plot_gaze_probe_kde(gaze_probe)  # prob map
            plot_gaze_rays(gaze_probe.density.mu, color='r', label='centers')
            plot_gaze_rays(rays, color='k', label='random')

        # with GazeProbePlotHelper(os.path.join(save_path, f'{name}_{i:04d}_gaze_pred.png')):
        with GazeProbePlotHelper():
            plt.title(f"Predicted gaze pdf [nearest interpolation from centers]")
            plot_gaze_rays(gaze_probe.density.mu, color='r', label='centers')
            plot_gaze_rays(rays, color='k', label='random')
            plot_gaze_rays_p(rays, prob)

        with GazeProbePlotHelper():
            plt.title(f"Predicted smoothed gaze pdf [nearest interpolation from centers]")
            plot_gaze_rays(gaze_probe.density.mu, color='r', label='centers')
            plot_gaze_rays(rays, color='k', label='random')
            plot_gaze_rays_p(rays, prob_smooth)

        plt.show()
