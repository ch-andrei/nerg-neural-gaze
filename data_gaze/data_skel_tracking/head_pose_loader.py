import os
import pathlib
from typing import List, Tuple
import pickle as pkl

import numpy as np
from matplotlib.collections import PatchCollection

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common.vector_math import normalize
from common.logging import log
from data_gaze.gaze_probe import GazeProbe, plot_gaze_probe_data, GazeProbePlotHelper, plot_gaze_probe_kde

#######################################################################################################################
# save preloaded data to a pkl

GAZE_DATASET_PATH = "./data_gaze/data_skel_tracking"
GAZE_DATA_FILE = "2023-11-01.csv"  # this file contains all the tracking data

GAZE_DATA_PKL_PATH = f"{GAZE_DATASET_PATH}/gaze_data.pkl"

# pickled dataset dict fields
NUM_TOTAL_LINES_FIELD = "num_total_lines"
DATA_FIELD = "data"

# DEBUG = True
DEBUG = False

#######################################################################################################################

# NOTE on convenience store gaze data:
# ro.shape = N, 3
# ro x min max 0.159 17.0565
# ro y min max -1.6644999999999999 7.444000000000001
# ro z min max 0.177 2.1870000000000003
# storage for (num_rays, ro, rd)
__gaze_data = None

# filter for gaze data; gaze rays that originate outside of these ranges are removed
GAZE_DATA_FILTER_XYZ_RANGE_ISLAND = 6.0, 10.5, 2.0, 6.5, 0.0, 2.5  # ~rectangle around the center isle

GAZE_DATA_FILTER_XYZ_RANGE = GAZE_DATA_FILTER_XYZ_RANGE_ISLAND

#######################################################################################################################
# matrix transformations for the central Isle to convert between physical store pos and pos given by COLMAP
# in Blender, run bpy.context.object.matrix_world when object is selected
# TODO: this comes from manual alignment... implement a better way to align nerf/gaze coords

# new (long video)
TRANSFORM_NERF2GAZE = np.array(
    (
        (-0.9146806597709656, -0.03410607948899269, 0.9522264003753662, 8.017474174499512),
        (0.9525607824325562, -0.06450771540403366, 0.9126914143562317, 4.220166206359863),
        (0.022938668727874756, 1.3187928199768066, 0.06926967203617096, 1.0269443988800049),
        (0.0, 0.0, 0.0, 1.0)  # since these are always 0,0,0,1, we can separately apply rotation/scale and translation
    )
)
TRANSFORM_GAZE2NERF = np.linalg.inv(TRANSFORM_NERF2GAZE)

TRANSFORM_NERF2GAZE_ROT_SCALE = TRANSFORM_NERF2GAZE[:3, :3]  # rotation and scaling
TRANSFORM_NERF2GAZE_XYZ = TRANSFORM_NERF2GAZE[:3, 3].reshape(3, 1)  # offset

TRANSFORM_GAZE2NERF_ROT_SCALE = TRANSFORM_GAZE2NERF[:3, :3]  # rotation and scaling
TRANSFORM_GAZE2NERF_XYZ = TRANSFORM_GAZE2NERF[:3, 3].reshape(3, 1)  # offset


def pos_nerf2gaze(v, offset=True):
    # input of shape 3,N
    v = TRANSFORM_NERF2GAZE_ROT_SCALE @ v
    if offset:
        v = v + TRANSFORM_NERF2GAZE_XYZ
    return v


def pos_gaze2nerf(v, offset=True):
    # input of shape 3,N
    v = TRANSFORM_GAZE2NERF_ROT_SCALE @ v
    if offset:
        v = v + TRANSFORM_GAZE2NERF_XYZ
    return v


def dir_nerf2gaze(d, norm=True):
    # input of shape 3,N
    d = pos_nerf2gaze(d, offset=False)  # directions only need to be rotated
    if norm:
        d = normalize(d, axis=0)
    return d


def dir_gaze2nerf(d, norm=True):
    # input of shape 3,N
    d = pos_gaze2nerf(d, offset=False)  # directions only need to be rotated
    if norm:
        d = normalize(d, axis=0)
    return d


#######################################################################################################################


def list_of_floats_from_string(s: str) -> List[float]:
    return [float(x) for x in s[1:-1].split(", ")]


def load_gaze_data_from_path(path: pathlib.Path, num_lines=-1) -> List[Tuple[str, int, List[float]]]:
    with path.open("rt") as infile:
        data_list = infile.read().split("\n")[:-1]

    _data_list = [line.split("|") for line in data_list]
    total_lines = len(_data_list)
    log(f"Gaze data file number of lines: {total_lines}")
    data_list = []
    for num_line, line in enumerate(_data_list):
        if num_line % 10000 == 0:
            print(f'\rLoading gaze data {num_line}/{total_lines}', end="")
        if 0 < num_lines < num_line:
            print(f"GazeDataset debug, stopping at {num_lines} lines.")
            break
        try:
            data_list.append((line[0], int(line[1]), list_of_floats_from_string(line[2])))
        except Exception as e:
            log(f"Processing line caused exception: {num_line} {e}")

    return data_list, num_line


def load_gaze_data(gaze_data_filter=GAZE_DATA_FILTER_XYZ_RANGE, plot_data=False, load_data_lines=-1, load_pkl=True, save_pkl=True):
    global __gaze_data

    # check if need to load
    if __gaze_data is None:
        if load_pkl:
            if os.path.isfile(GAZE_DATA_PKL_PATH):
                with open(GAZE_DATA_PKL_PATH, 'rb') as data_file:
                    data = pkl.load(data_file)
                    __gaze_data = data[DATA_FIELD]
                    log(f"Read dataset pkl (num_lines={data[NUM_TOTAL_LINES_FIELD]}).")
                    return __gaze_data
            else:
                log(f"Could not read dataset pkl. Recomputing.")

        csv_path = f"{GAZE_DATASET_PATH}/{GAZE_DATA_FILE}"
        path = pathlib.Path(csv_path)

        data_list, num_lines_total = load_gaze_data_from_path(path, num_lines=load_data_lines)

        pose_info = np.array([data[2] for data in data_list])
        left_ear = pose_info[:, 6:9]
        right_ear = pose_info[:, 39:42]
        nose = pose_info[:, 33:36]
        ro = (left_ear + right_ear) / 2

        # check for nans
        valid_ids = np.logical_not(np.logical_or(
            np.any(np.isnan(nose), axis=1),
            np.any(np.isnan(ro), axis=1)
        ))
        nose = nose[valid_ids]
        ro = ro[valid_ids]

        if plot_data:
            def plot_points(p, color_letter='b', num_points=100000):
                num_points = min(num_points, p.shape[0])
                plot_idx = np.random.choice(p.shape[0], size=num_points, replace=False)
                p = p[plot_idx]
                plt.plot(p[..., 0], p[..., 1], f'{color_letter},')
                plt.draw()

            # plot available gaze ray centers
            fig, ax = plt.subplots(1)
            plot_points(ro, 'b', 100000)

        # apply position filter
        xmin, xmax, ymin, ymax, zmin, zmax = gaze_data_filter  # xy filter min/max
        valid_ids = np.logical_and.reduce([
            np.logical_and(xmin < ro[..., 0], ro[..., 0] < xmax),
            np.logical_and(ymin < ro[..., 1], ro[..., 1] < ymax),
            np.logical_and(zmin < ro[..., 2], ro[..., 2] < zmax)
        ])
        nose = nose[valid_ids]
        ro = ro[valid_ids]
        num_rays = np.sum(valid_ids)

        if plot_data:
            ## plot selected gaze ray centers
            selected_color = 'r'
            # plot_points(ro, selected_color, 10000)
            selected_area = [Rectangle((xmin, ymin), xmax - xmin, ymax - ymin)]
            # Create patch collection with specified colour/alpha
            pc = PatchCollection(selected_area, facecolor=None, alpha=0.5, edgecolor=selected_color)
            ax.add_collection(pc)
            ax.axis('equal')
            plt.draw()

        rd = normalize(nose - ro)

        __gaze_data = num_rays, ro, rd  # N, (N,3), (N,3)

        if save_pkl:
            with open(GAZE_DATA_PKL_PATH, 'wb') as data_file:
                log("Saving dataset pkl.")
                data = {
                    NUM_TOTAL_LINES_FIELD: num_lines_total,
                    DATA_FIELD: __gaze_data
                }
                pkl.dump(data, data_file)

    if plot_data:
        plt.draw()

    return __gaze_data


# plots positions of gaze probes
def plot_gaze_probe_centers(gaze_probes: list, tag="", save_path=None, nerf_coords=False):
    if len(gaze_probes) == 0:
        log("Plotting gaze probes but the list of spheres is empty.")
        return

    tag = f"[{tag}] " if tag else ""

    centers = []
    counts = []
    for gaze_probe in gaze_probes:  # type: GazeProbe
        if nerf_coords:
            centers.append(pos_gaze2nerf(gaze_probe.ro))
        else:
            centers.append(gaze_probe.ro)
        counts.append(gaze_probe.num_rays)

    centers = np.array(centers, float)
    N = centers.shape[0]
    counts = np.array(counts, float)
    colors = counts - counts.min()
    colors = colors / colors.max()
    colors3 = np.zeros((N, 3), float)
    colors3[:, 0] = colors

    fig = plt.figure(dpi=300, figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')
    plt.title(f"{tag}Gaze probes distribution")

    print("Plotting gaze probes...")
    print('nerf_coords', nerf_coords)
    print('centers', centers.shape)
    print('colors', colors.shape)

    if nerf_coords:
        xs = centers[:, 0]
        ys = centers[:, 1]
        zs = centers[:, 2]
    else:
        xs = centers[:, 0]
        ys = centers[:, 1]
        zs = centers[:, 2]

    plt.xlabel('x')
    plt.ylabel('y')

    ax.scatter(xs, ys, zs, marker="o", s=2, c=colors3)

    if nerf_coords:
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        ax.set_zlim(-2, 2)
    else:
        plt.xlim(*GAZE_DATA_FILTER_XYZ_RANGE_ISLAND[0:2])
        plt.ylim(*GAZE_DATA_FILTER_XYZ_RANGE_ISLAND[2:4])
        ax.set_zlim(*GAZE_DATA_FILTER_XYZ_RANGE_ISLAND[4:6])

    plt.draw()

    if save_path:
        plt.savefig(save_path)
