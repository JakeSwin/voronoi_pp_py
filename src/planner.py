import jax.numpy as jnp
import numpy as np
import rerun as rr

from scipy.spatial import Delaunay

from src.uav import UAV
from src.gps.numpyro_gp import GP
from src.util import find_neighbours_delaunay

def generate_candidate_paths(delaunay, starting_idx, horizon_size):
    indptr, indices = delaunay.vertex_neighbor_vertices
    candidate_paths = [[int(starting_idx)]]

    for i in range(horizon_size):
        new_candidate_paths = []
        for cpath in candidate_paths:
            last_idx = cpath[-1]
            neighbours = indices[indptr[last_idx] : indptr[last_idx + 1]]
            for n in neighbours:
                if n not in cpath:
                    new_candidate_paths.append(cpath + [int(n)])
        candidate_paths = new_candidate_paths

    return candidate_paths

class PlannerClosest:
    def __init__(self):
        self.num_steps = 0
        self.tri = None
        self.neighbour_points = None
        self.current_pos = None
        self.current_idx = None

    def step(self, uav: UAV, index_map, centroids):
        self.current_pos = uav.pos

        texture_width = index_map.shape[1]
        texture_height = index_map.shape[0]
        current_x = jnp.clip(jnp.round(self.current_pos[0]), 0, texture_width).astype(jnp.int32)
        current_y = jnp.clip(jnp.round(self.current_pos[1]), 0, texture_height).astype(jnp.int32)
        self.current_idx = index_map[current_y, current_x]

        rows_with_nan = jnp.any(jnp.isnan(centroids), axis=1)
        clean_centroids = centroids[~rows_with_nan]
        self.tri = Delaunay(clean_centroids)
        neighbour_idxs = find_neighbours_delaunay(self.tri, self.current_idx)

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
        if self.tri == None or self.neighbour_points == None or self.current_pos == None or self.current_idx == None:
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

        candidate_path_idxs = generate_candidate_paths(self.tri, self.current_idx, 3)
        candidate_paths = centroids[candidate_path_idxs].astype(np.float32)

        rr.log("Voronoi/Candidates", rr.LineStrips2D(np.asarray(candidate_paths)))

def get_information_gain(new_path, gp, X_train, Y_train):
    mu_prior, cov_prior = gp.predict(X_train, Y_train, new_path)

    # Calculate prior entropy
    d = new_path.shape[0]
    H_prior = 0.5 * jnp.log((2 * jnp.pi * jnp.e) ** d * jnp.linalg.det(cov_prior))

    X_train_updated = jnp.concatenate([X_train, new_path])
    Y_train_updated = jnp.concatenate([Y_train, mu_prior])

    _, cov_post = gp.predict(X_train_updated, Y_train_updated, new_path)

    # Calculate posterior entropy
    H_post = 0.5 * jnp.log((2 * jnp.pi * jnp.e) ** d * jnp.linalg.det(cov_post))

    info_gain = H_prior - H_post

    return info_gain

class PlannerRH:
    def __init__(self, horizon_size = 3):
        self.num_steps = 0
        self.tri = None
        self.current_pos = None
        self.current_idx = None
        self.horizon_size = horizon_size

    def step(self, uav: UAV, gp: GP, normalisation_func, index_map, centroids):
        self.current_pos = uav.pos

        texture_width = index_map.shape[1]
        texture_height = index_map.shape[0]
        current_x = jnp.clip(jnp.round(self.current_pos[0]), 0, texture_width).astype(jnp.int32)
        current_y = jnp.clip(jnp.round(self.current_pos[1]), 0, texture_height).astype(jnp.int32)
        self.current_idx = index_map[current_y, current_x]

        rows_with_nan = jnp.any(jnp.isnan(centroids), axis=1)
        clean_centroids = jnp.nan_to_num(centroids, 0)
        self.tri = Delaunay(clean_centroids)

        candidate_paths_idxs = generate_candidate_paths(self.tri, self.current_idx, self.horizon_size)
        candidate_paths_idxs = np.array(candidate_paths_idxs)[:, 1:]
        candidate_paths = clean_centroids[candidate_paths_idxs]
        candidate_paths_normalised, _, _  = normalisation_func(candidate_paths)

        current_path = uav.get_full_path()
        normalised_current_path, _, _  = normalisation_func(current_path)

        eval_list = [get_information_gain(path, gp, normalised_current_path, uav.samples) for path in candidate_paths_normalised]
        # Sort in desc order (larger is better)
        sorted_eval_list = np.argsort(eval_list)[::-1]

        # Need to remove points that move to nan centroids or to areas already pathed to

        # Take only the first point from the best path
        closest_point = candidate_paths[sorted_eval_list[0], 0]

        uav.move(closest_point)
        uav.log()

        self.num_steps += 1

    def log(self):
        if self.tri == None or self.current_pos == None or self.current_idx == None:
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

        candidate_path_idxs = generate_candidate_paths(self.tri, self.current_idx, 3)
        candidate_paths = centroids[candidate_path_idxs].astype(np.float32)

        rr.log("Voronoi/Candidates", rr.LineStrips2D(np.asarray(candidate_paths)))
