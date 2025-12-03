import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt

from data_gaze.gaze_probe import GazeProbe, plot_gaze_probe_data
from data_gaze.data_skel_tracking.head_pose_loader import load_gaze_data, dir_gaze2nerf, pos_gaze2nerf

from common.numpy_utils import TemporaryNumpySeed
from common.logging import LogOnTaskComplete
from common.mp_task import process_multithreaded


########################################################################################################################

# gaze queries can be based on
# 1. Radius = get all gaze rays within radius R of the query position
# 2. Count = get N closest gaze rays for the query position
GAZE_QUERY_TYPE_RADIUS = 0
GAZE_QUERY_TYPE_COUNT = 1


class GazeDataset(Dataset):
    def __init__(
            self,

            device,

            gaze_query_type=GAZE_QUERY_TYPE_RADIUS,  # query gaze rays by radius or by count
            gaze_query_radius=0.1,  # gaze probes collect gaze rays within a given radius (in meters)
            gaze_query_count=1024,  # if gaze_query_type==GAZE_QUERY_TYPE_COUNT, number of gaze rays to collect

            gaze_query_count_min=8,  # minimum allowed number of gaze rays per gaze probe
            gaze_query_count_max=4096,  # maximum allowed number of gaze rays per gaze probe

            preload_gaze_probes=True,
            preload_gaze_rays=True,
            # gaze_probe_count=1000,  #
            gaze_probe_count=4096,  #
            gaze_probe_seed=False,  # if None, will not use custom seed. Only applies when preloading gaze probes

            num_gaze_rays=4096*4,

            # debugging
            gaze_debug_load_lines=-1,
            # gaze_debug_load_lines=100000,  # load fewer lines from the data file

            gaze_debug_plot_data=False,
            gaze_probes_plot=False,

            return_gaze_probe=False,

            # gaze_probe_plot=True
    ):

        self.device = device

        self.gaze_query_type = gaze_query_type
        self.gaze_query_radius = gaze_query_radius
        self.gaze_query_count = gaze_query_count
        self.gaze_query_count_min = gaze_query_count_min
        self.gaze_query_count_max = gaze_query_count_max

        if not (gaze_query_count_min < gaze_query_count < gaze_query_count_max):
            raise ValueError("Incorrect gaze query count configuration.")

        self.num_gaze_rays = num_gaze_rays

        with LogOnTaskComplete("Loading gaze data"):
            num_rays, ro, rd = load_gaze_data(plot_data=gaze_debug_plot_data, load_data_lines=gaze_debug_load_lines)

        with LogOnTaskComplete("Building gaze KDTree"):
            gaze_tree = KDTree(ro)

        self.return_gaze_probe = return_gaze_probe

        self.preload_gaze_probes = preload_gaze_probes
        self.preload_gaze_rays = preload_gaze_rays  # can be disabled later

        self.gaze_probes = []
        self.gaze_probe_count = gaze_probe_count
        if preload_gaze_probes:
            with TemporaryNumpySeed(seed=gaze_probe_seed):  # NOTE: this does not change global seed
                # precompute gaze probes, dont save ro/rd/kdtree
                with LogOnTaskComplete("Preloading gaze probes"):
                    self.gaze_probes = self.sample_gaze_probes(
                        gaze_probe_count, num_rays, ro, rd, gaze_tree,
                        preload_gaze_rays=num_gaze_rays if preload_gaze_rays else -1,
                    )

                if gaze_probes_plot:
                    for gaze_probe in self.gaze_probes[:10]:
                        plot_gaze_probe_data(gaze_probe)
                    plt.show()

                self.gaze_probe_count = len(self.gaze_probes)

        else:
            # save data for later sampling of gaze_probes
            self.gaze_tree = gaze_tree
            self.ro = ro
            self.rd = rd
            self.num_rays = num_rays

            # raise NotImplementedError()  # TODO implement non-recomputed sampling

    def query_gaze_tree(self, query_points, gaze_tree=None):
        if gaze_tree is None:
            gaze_tree = self.gaze_tree

        if self.gaze_query_type == GAZE_QUERY_TYPE_RADIUS:
            # query by search radius
            inds_near, dists = gaze_tree.query_radius(query_points, r=self.gaze_query_radius, return_distance=True, sort_results=True)

        elif self.gaze_query_type == GAZE_QUERY_TYPE_COUNT:
            # query by desired count
            dists, inds_near = gaze_tree.query(query_points, k=self.gaze_query_count, return_distance=True)

        else:
            raise ValueError()

        return dists, inds_near

    @staticmethod
    def try_get_gaze_probes_task(
            i, gaze_query_type, gaze_query_radius, gaze_query_count_min, gaze_query_count_max,
            inds_near, dists, ro, rd, preload_gaze_rays, reject_samples):

        inds_near_i = inds_near[i]
        dists_i = dists[i].flatten()

        if reject_samples and gaze_query_type == GAZE_QUERY_TYPE_COUNT:
            # when querying tree by count, reject rays that are too far
            good_inds = dists_i < gaze_query_radius
            dists_i = dists_i[good_inds]
            inds_near_i = inds_near_i[good_inds]

        dists_i = dists_i[:gaze_query_count_max]

        # reject if too few gaze rays
        num_rays = len(inds_near_i)
        if reject_samples and num_rays < gaze_query_count_min:
            return None  # too few rays

        ro_near = ro[inds_near_i][:gaze_query_count_max].transpose()  # from N,3 to 3,N used by KDE
        rd_near = rd[inds_near_i][:gaze_query_count_max].transpose()  # from N,3 to 3,N used by KDE

        gaze_probe = GazeProbe(ro_near, rd_near, dists_i, preload_num_gaze_rays=preload_gaze_rays)

        return gaze_probe

    def try_get_gaze_probes_multiprocessed(
            self, num_queries, inds_near, dists, ro, rd,
            preload_gaze_rays=-1, reject_samples=False, multiprocess=True
    ):
        # when multiprocess=True, use multiple threads to process gaze probes.
        # Note: KDTree is already multi-threaded but this speeds up the rest.
        # add 1 process for every power of 10 for num_queries (up to 4 processes)
        # ex: log10(10)=1 process, log10(10000)=4 processes
        num_processes = min(4, max(1, int(np.log10(num_queries)))) if multiprocess else 1
        items = process_multithreaded(
            list(range(num_queries)),
            GazeDataset.try_get_gaze_probes_task,
            task_args=[
                self.gaze_query_type, self.gaze_query_radius, self.gaze_query_count_min, self.gaze_query_count_max,
                inds_near, dists, ro, rd, preload_gaze_rays, reject_samples
            ],
            num_processes=num_processes, raise_exceptions=True
        )
        gaze_probes = [item for item in items.values() if item is not None]  # remove Nones

        return gaze_probes

    def get_gaze_probes_at(self, query_points):
        # query_points N,3
        num_query_points = query_points.shape[0]

        dists, inds_near = self.query_gaze_tree(query_points)

        gaze_probes = self.try_get_gaze_probes_multiprocessed(
            num_query_points, inds_near, dists, self.ro, self.rd, reject_samples=False)

        return gaze_probes

    def sample_gaze_probes(self, num_probes, num_rays, ro, rd, gaze_tree, preload_gaze_rays=-1, max_iterations=100):
        gaze_probes = []
        iteration = 0
        while len(gaze_probes) < num_probes and iteration < max_iterations:
            num_probes_remain = num_probes - len(gaze_probes)

            query_points = np.random.randint(num_rays, size=max(1, num_probes_remain))
            query_points = ro[query_points]

            dists, inds_near = self.query_gaze_tree(query_points, gaze_tree)

            for gaze_probe in self.try_get_gaze_probes_multiprocessed(
                    num_probes_remain, inds_near, dists, ro, rd, preload_gaze_rays, reject_samples=True):
                gaze_probes.append(gaze_probe)

            iteration += 1

        # save sampled gaze probes
        for gaze_probe in gaze_probes:
            if self.gaze_probe_count <= len(self.gaze_probes):
                self.gaze_probes.pop(0)  # remove first
                self.gaze_probes.append(gaze_probe)  # add last

        return gaze_probes

    @torch.cuda.amp.autocast(enabled=False)
    # @torch.no_grad()
    def get_gaze_rays(self, gaze_probe: GazeProbe, allow_preloaded=True):
        rays_d, log_prob = gaze_probe.sample_gaze_rays(self.num_gaze_rays, allow_preloaded=allow_preloaded)
        rays_d = dir_gaze2nerf(rays_d).transpose()  # from 3,N to N,3; rotate but not translate
        rays_d = torch.from_numpy(rays_d)

        num_gaze_rays = rays_d.shape[0]  # this might not be self.num_gaze_rays if precomputed

        rays_o = pos_gaze2nerf(gaze_probe.ro).transpose()  # from 3,1 to 1,3 and transform to nerf coords
        rays_o = torch.from_numpy(rays_o).expand(num_gaze_rays, 3)

        gaze_prob = np.exp(log_prob)
        gaze_prob = torch.from_numpy(gaze_prob)

        return rays_o, rays_d, gaze_prob

    def __len__(self):
        return len(self.gaze_probes) if self.preload_gaze_probes else self.gaze_probe_count

    # @torch.no_grad()
    def collate(self, indices):
        idx = indices[0]  # indices is a list but batch size is always 1

        if self.preload_gaze_probes:
            # get a precomputed gaze probe
            gaze_probe = self.gaze_probes[idx]
        else:
            # sample a random gaze probe
            gaze_probe = self.sample_gaze_probes(
                1, self.num_rays, self.ro, self.rd, self.gaze_tree, preload_gaze_rays=self.num_gaze_rays)[0]

        rays_o, rays_d, gaze_prob = self.get_gaze_rays(gaze_probe, allow_preloaded=self.preload_gaze_rays)

        data = dict()
        data["rays_o"] = rays_o.to(self.device, dtype=torch.float).view(1, -1, 3)  # N,3
        data["rays_d"] = rays_d.to(self.device, dtype=torch.float).view(1, -1, 3)  # N,3
        data['gaze_prob'] = gaze_prob.to(self.device, dtype=torch.float).view(1, -1)  # N
        # our "image" is of shape (1,num_gaze_rays)
        data["H"] = 1
        data["W"] = self.num_gaze_rays

        if self.return_gaze_probe:
            data["gaze_probe"] = gaze_probe

        return data

    def dataloader(self, num_workers=8):
        size = len(self)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=True, num_workers=num_workers)
        loader.has_gt = True
        return loader
