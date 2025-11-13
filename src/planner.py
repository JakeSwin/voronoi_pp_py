import jax.numpy as jnp

from scipy.spatial import Delaunay

from src.uav import UAV
from src.util import find_neighbours_delaunay

class Planner:
    def __init__(self):
        self.num_steps = 0

    def step(self, uav: UAV, index_map, centroids):
        texture_width = index_map.shape[1]
        texture_height = index_map.shape[0]
        current_x = jnp.clip(jnp.round(uav.pos[0]), 0, texture_width).astype(jnp.int32)
        current_y = jnp.clip(jnp.round(uav.pos[1]), 0, texture_height).astype(jnp.int32)
        current_cell_index = index_map[current_y, current_x]

        rows_with_nan = jnp.any(jnp.isnan(centroids), axis=1)
        clean_centroids = centroids[~rows_with_nan]
        tri = Delaunay(clean_centroids)
        neighbour_idxs = find_neighbours_delaunay(tri, current_cell_index)

        if uav.path != None:
            path_xs = jnp.clip(jnp.round(uav.path[:, 0]), 0, texture_width).astype(jnp.int32)
            path_ys = jnp.clip(jnp.round(uav.path[:, 1]), 0, texture_height).astype(jnp.int32)
            path_cell_idxs = index_map[path_ys, path_xs]
            neighbour_idxs = neighbour_idxs[~jnp.isin(neighbour_idxs, path_cell_idxs)]

        neighbour_points = centroids[neighbour_idxs]

        dists = jnp.linalg.norm(neighbour_points - uav.pos, axis=-1)
        closest_point = neighbour_points[dists.argmin()]

        uav.move(closest_point)
        uav.log()

        self.num_steps += 1
