import jax.numpy as jnp
import numpy as np
import rerun as rr

from scipy.spatial import Delaunay

from src.uav import UAV
from src.util import find_neighbours_delaunay

class Planner:
    def __init__(self):
        self.num_steps = 0
        self.tri = None
        self.neighbour_points = None
        self.current_pos = None

    def step(self, uav: UAV, index_map, centroids):
        self.current_pos = uav.pos

        texture_width = index_map.shape[1]
        texture_height = index_map.shape[0]
        current_x = jnp.clip(jnp.round(self.current_pos[0]), 0, texture_width).astype(jnp.int32)
        current_y = jnp.clip(jnp.round(self.current_pos[1]), 0, texture_height).astype(jnp.int32)
        current_cell_index = index_map[current_y, current_x]

        rows_with_nan = jnp.any(jnp.isnan(centroids), axis=1)
        clean_centroids = centroids[~rows_with_nan]
        self.tri = Delaunay(clean_centroids)
        neighbour_idxs = find_neighbours_delaunay(self.tri, current_cell_index)

        if uav.path != None:
            path_xs = jnp.clip(jnp.round(uav.path[:, 0]), 0, texture_width).astype(jnp.int32)
            path_ys = jnp.clip(jnp.round(uav.path[:, 1]), 0, texture_height).astype(jnp.int32)
            path_cell_idxs = index_map[path_ys, path_xs]
            neighbour_idxs = neighbour_idxs[~jnp.isin(neighbour_idxs, path_cell_idxs)]

        self.neighbour_points = centroids[neighbour_idxs]

        dists = jnp.linalg.norm(self.neighbour_points - self.current_pos, axis=-1)
        closest_point = self.neighbour_points[dists.argmin()]

        uav.move(closest_point)
        uav.log()

        self.num_steps += 1

    def log(self):
        if self.tri == None or self.neighbour_points == None or self.current_pos == None:
            return

        edges = set()
        for simplex in self.tri.simplices:
            # Each triangle contributes three edges
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edges.add(edge)

        delaunay_idxs = np.array(list(edges))
        centroids = np.asarray(self.tri.points)
        delaunay_lines = centroids[delaunay_idxs].astype(np.float32)

        rr.log("Voronoi/Delaunay", rr.LineStrips2D(delaunay_lines))

        num_neighbour_points = self.neighbour_points.shape[0]
        current_pos_repeat = jnp.repeat(self.current_pos[None, :], num_neighbour_points, axis=0)
        neighbour_lines = jnp.stack((current_pos_repeat, self.neighbour_points), axis=1)

        rr.log("Voronoi/Neighbours", rr.LineStrips2D(np.asarray(neighbour_lines)))
