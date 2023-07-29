import numpy as np
import cc3d
import scipy
from typing import List

import torch
#from nerfacc import OccupancyGrid
from nerfacc import OccGridEstimator


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def viz_top_k_components(components):
    # TODO: Maybe remove
    import pyvista as pv

    N_components = len(components)
    blocks = pv.MultiBlock()
    for segid in range(N_components):
        blocks[str(segid)] = components[segid] * 100

    pl = pv.Plotter(notebook=False)
    pl.add_volume(blocks, multi_colors=True)
    pl.show()


def extract_top_k_connected_component(density_grid: np.ndarray,
                                      threshold: float = 0.6,
                                      sigma_thinning: float = 1,
                                      sigma_erosion: float = 2,
                                      K=1
                                      ) -> List[np.ndarray]:
    """

    Args:
      density_grid: density score grid from NeRF-like model; shape: [dim, dim, dim]
      threshold: after applying sigmoid to densities, remove all cells lower than threshold
      sigma_thinning: standard deviation of gaussian kernel used to blur density grid, this should be able to remove thin connections
      sigma_erosion: standard deviation of gaussian kernel used to enlarge largest connected component (others are not enlarged)
      K: number of connected components to return

    Returns: List of binary masks of largest connected components in density_grid; shapes are [dim, dim, dim]

    """

    # maybe applying a sigmoid layer on the density scores is nicer
    # clamp_max = clamp
    # density_grid = np.clip(density_grid, 0, clamp_max)
    # density_grid = (density_grid*(255/clamp_max)).astype('uint8')

    density_grid = sigmoid(density_grid)
    density_grid = ((density_grid - 0.5) * 2 * (255)).astype(
        np.uint8)  # rescaling to 255 and casting for commented pyvista volume rendering

    # apply gaussian filter to "break" narrow connections
    density_grid = scipy.ndimage.gaussian_filter(density_grid, sigma=sigma_thinning)

    # remove all densities below a threshold

    density_grid[density_grid < 255 * threshold] = 0
    density_grid[density_grid >= 255 * threshold] = 1

    # import pyvista as pv
    # print('[LOG] Plotting density grid after filtering and thresholding')
    # grid = pv.UniformGrid()
    # grid.dimensions = np.array(density_grid.shape) + 1
    # grid.cell_data["values"] = density_grid.flatten(order="F")
    # pl = pv.Plotter()
    # pl.add_volume(grid, cmap="coolwarm")
    # pl.show()

    # extract largest connected component
    connectivity = 6
    labels_out, N = cc3d.largest_k(
        density_grid, k=K,
        connectivity=connectivity, delta=0,
        return_N=True,
    )

    ccs = []
    for k in range(1, K + 1):
        curr_cc = labels_out == k

        # for largest component erode grid to enlarge active region
        if k == K:
            curr_cc = scipy.ndimage.gaussian_filter(curr_cc * 100, sigma=sigma_erosion)

        out = np.zeros_like(curr_cc)
        out[curr_cc > 0] = 1
        ccs.append(out)

        # print('[LOG] Plotting final active region')
        # pl = pv.Plotter()
        # pl.add_volume(out*100,  multi_colors=True)#, cmap="coolwarm")
        # pl.show()

    return ccs


def filter_occupancy_grid(occupancy_grid: OccGridEstimator,
                          threshold: float = 0.6,
                          sigma_thinning: float = 1,
                          sigma_erosion: float = 5):
    """
    Filters the provided occupancy grid such that it only contains voxels that belong to the largest connected component
    of active voxels in the grid.
    This removes isolated islands of floaters for better visual quality.

    Args:
        occupancy_grid: the occupancy grid to filter
        threshold: sigmoidal densities below that threshold will be truncated
        sigma_thinning: removes thin connections between larger components to separate smaller floaters
        sigma_erosion: enlarges the final component in the end to not crop away too much
    """

    resolution = occupancy_grid.resolution
    try:
        iter(resolution)
    except TypeError:
        # If resolution is not iterable, it probably was a single number
        resolution = [resolution, resolution, resolution]

    occupancy_grid_densities = occupancy_grid.occs
    occupancy_grid_densities = occupancy_grid_densities.reshape(*resolution)
    occupancy_grid_densities = occupancy_grid_densities.cpu().numpy()

    largest_connected_component = \
        extract_top_k_connected_component(occupancy_grid_densities,
                                          threshold=threshold,
                                          sigma_thinning=sigma_thinning,
                                          sigma_erosion=sigma_erosion)[0]

    filtered_occupancy_grid = largest_connected_component > 0  # Make binary
    filtered_occupancy_grid = torch.tensor(filtered_occupancy_grid,
                                           device=occupancy_grid.device,
                                           dtype=occupancy_grid.binaries.dtype)
    occupancy_grid.binaries[0] = occupancy_grid.binaries[0] & filtered_occupancy_grid
