import jax
import jax.numpy as jnp

from jaxtyping import Array
from src.constants import NUM_SEEDS, LOWER_THRESHOLD, UPPER_THRESHOLD

offset_vectors = jnp.array(
    [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
)  # For jump, scale by offset


class Voronoi:
    def __init__(self, size: int, seeds: Array):
        self.setup(size, seeds)

    def setup(self, size: int, seeds: Array):
        seeds = jnp.round(jnp.array(seeds)).astype(jnp.int16)
        self.size = size
        self.numseeds = seeds.shape[0]

        with jax.default_device(jax.devices("cpu")[0]):
            # arr = jnp.zeros((size, size, 2))
            # self.seed_map = arr.at[seeds[:, 0], seeds[:, 1]].set(seeds)
            arr = jnp.zeros((size, size, 2, 2), dtype=seeds.dtype)
            seed_values = jnp.tile(seeds[:, None, :], (1, 2, 1))
            self.seed_map = arr.at[seeds[:, 0], seeds[:, 1]].set(seed_values)

            max_exp = int(jnp.floor(jnp.log2(size // 2)))
            # JFA+2 variant for more accurate border
            self.jfa_offsets = [(size // 2) // (2**i) for i in range(max_exp + 1)] + [
                1,
                1,
            ]

    @staticmethod
    @jax.jit
    def _make_seeded_arr(size: int, seeds: Array) -> Array:
        arr = jnp.zeros((size, size, 2))
        return arr.at[seeds[:, 0], seeds[:, 1]].set(seeds)

    @staticmethod
    @jax.jit
    def _jfa_step(arr: Array, offset: int):
        size = arr.shape[0]

        def step(i, j):
            pos = jnp.array([i, j])
            shifts = offset_vectors * offset
            candidates = pos + shifts
            candidates = jnp.clip(candidates, 0, size - 1)
            info = arr[candidates[:, 0], candidates[:, 1]]
            all_sites = info.reshape(-1, 2)
            dists = jnp.linalg.norm(all_sites - pos, axis=-1)
            # argmin = jnp.argmin(dists)
            # _, idxs = jax.lax.top_k(-dists, 2)
            # first = all_sites[idxs[0]]
            # second = all_sites[idxs[1]]
            idxs = jnp.argsort(dists)
            first = all_sites[idxs[0]]  # Closest

            # Create mask: True for candidates not equal to first
            mask = ~jnp.all(all_sites == first, axis=1)
            # Use a large number for masked-out values so they're not chosen
            dists_masked = jnp.where(mask, dists, jnp.inf)
            # Now pick the argmin among these (second unique site)
            idx2 = jnp.argmin(dists_masked)
            second = all_sites[idx2]
            return jnp.stack([first, second], axis=0)

        vmapped = jax.vmap(jax.vmap(step, in_axes=(None, 0)), in_axes=(0, None))
        grid = jnp.arange(size)
        return vmapped(grid, grid)

    def jfa(self):
        arr = self.seed_map  # Size: (N, N, 2, 2)
        for offset in self.jfa_offsets:
            arr = self._jfa_step(arr, offset)
        size = arr.shape[0]
        grid_x, grid_y = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
        pos_grid = jnp.stack([grid_x, grid_y], axis=-1)  # shape: (N, N, 2)
        first_seed_grid = arr[..., 0, :]
        second_seed_grid = arr[..., 1, :]
        dist_first = jnp.linalg.norm(first_seed_grid - pos_grid, axis=-1)
        dist_second = jnp.linalg.norm(second_seed_grid - pos_grid, axis=-1)

        # Compute mask: True if first is closer, False otherwise
        mask = dist_first < dist_second

        # Select closest for each pixel
        closest = jnp.where(mask[..., None], first_seed_grid, second_seed_grid)
        second_closest = jnp.where(mask[..., None], second_seed_grid, first_seed_grid)

        # Stack so output shape is (N, N, 2, 2), order is [closest, second_closest]
        result = jnp.stack([closest, second_closest], axis=-2)
        return result

    @staticmethod
    @jax.jit
    def get_distance_transform(jfa_map: Array, dist_idx: int = 0):
        arr = jfa_map[:, :, dist_idx]
        size = jfa_map.shape[0]

        def step(i, j):
            pos = jnp.array([i, j])
            seed_pos = jfa_map[i, j]
            return jnp.linalg.norm(seed_pos - pos)

        vmapped = jax.vmap(jax.vmap(step, in_axes=(None, 0)), in_axes=(0, None))
        grid = jnp.arange(size)
        dist_transform = vmapped(grid, grid)

        # Set 1-pixel border to 0
        dist_transform = dist_transform.at[0:2, :].set(0)
        dist_transform = dist_transform.at[-2:, :].set(0)
        dist_transform = dist_transform.at[:, 0:2].set(0)
        dist_transform = dist_transform.at[:, -2:].set(0)

        return dist_transform

    @staticmethod
    @jax.jit
    def get_border_distance_transform(jfa_map: Array):
        # jfa_map shape: (H, W, 2, 2), e.g., two seeds for each pixel, each with 2D coordinate
        # jfa_map[..., 0, :] = closest seed, jfa_map[..., 1, :] = second closest seed

        H, W, _, _ = jfa_map.shape
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        pos = jnp.stack([yy, xx], axis=-1)

        a = jfa_map[..., 0, :]  # Closest seed coord
        b = jfa_map[..., 1, :]  # Second seed coord

        # Vector version of the perpendicular bisector formula
        ba = b - a
        pa = pos - a
        pb = pos - b
        numerator = jnp.abs(jnp.sum(ba * (2 * pos - a - b), axis=-1))
        denominator = 2 * jnp.linalg.norm(ba, axis=-1) + 1e-12  # avoid div-zero
        border_distance = numerator / denominator

        # Set 2-pixel border to 0
        border_distance = border_distance.at[0:2, :].set(0)
        border_distance = border_distance.at[-2:, :].set(0)
        border_distance = border_distance.at[:, 0:2].set(0)
        border_distance = border_distance.at[:, -2:].set(0)

        return border_distance

    @staticmethod
    def get_voro_centroids(index_map: Array, num_seeds: int) -> Array:
        # Build grid of coordinate indices
        grid = jnp.arange(index_map.shape[0])
        xx, yy = jnp.meshgrid(grid, grid)
        coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape (N, 2)

        flat_index_map = index_map.ravel()

        sum_x = jnp.bincount(flat_index_map, weights=coords[:, 0], length=num_seeds)
        sum_y = jnp.bincount(flat_index_map, weights=coords[:, 1], length=num_seeds)

        index_sum = jnp.bincount(flat_index_map, length=num_seeds)

        centroids_x = sum_x / index_sum
        centroids_y = sum_y / index_sum

        return jnp.stack([centroids_x, centroids_y], axis=1)

    @staticmethod
    def get_inscribing_circles(
        index_map: Array, distance_transform: Array, num_seeds: int
    ):
        # Build grid of coordinate indices
        grid = jnp.arange(index_map.shape[0])
        xx, yy = jnp.meshgrid(grid, grid)
        coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape (N, 2)

        flat_index_map = index_map.ravel()
        flat_dist_transform = distance_transform.ravel()
        max_masked_array = jax.ops.segment_max(
            flat_dist_transform, flat_index_map, num_seeds
        )
        seg_max_map = max_masked_array[flat_index_map]
        is_max = flat_dist_transform == seg_max_map
        indices = jnp.arange(flat_index_map.size)

        def argmax_group(seg_id):
            # Vectorized version
            possible = jnp.where(
                (flat_index_map == seg_id) & is_max, indices, flat_index_map.size
            )
            # Returns the first occurrence of the max
            return jnp.min(possible)

        # Use vmap to vectorize over segment IDs
        max_indices = jax.vmap(argmax_group)(jnp.arange(num_seeds))
        return coords[max_indices], max_masked_array

    @staticmethod
    def get_largest_extent(index_map: Array, distance_transform: Array, seeds: Array):
        # Gets the coordinate of the max value for distance transform from closest seed
        coords, max_masked_array = Voronoi.get_inscribing_circles(
            index_map, distance_transform, len(seeds)
        )

        seed_vectors = coords - seeds  # (num_seeds, 2)
        seed_norms = jnp.linalg.norm(
            seed_vectors, axis=1, keepdims=True
        )  # (num_seeds, 1)
        unit_vectors = seed_vectors / (
            seed_norms + 1e-8
        )  # Add epsilon to avoid division by zero
        return coords, unit_vectors

    @staticmethod
    def get_split(unit_vectors: Array, dists: Array, seeds: Array):
        lower_pos = seeds + (-dists[:, None] * unit_vectors)
        upper_pos = seeds + (dists[:, None] * unit_vectors)
        return jnp.stack([lower_pos, upper_pos], axis=1)

    @staticmethod
    def get_index_map(jfa_map: Array, seeds: Array):
        arr = jfa_map[:, :, 0]  # Only select closest seed value
        flat_coords = arr.reshape(-1, arr.shape[-1])  # shape (H*W, 2)
        seeds = jnp.round(jnp.array(seeds)).astype(jnp.int32)
        # Get each pixel's index in seeds (by matching coordinates for each pixel)
        # Broadcasting: (H*W, 1, 2) == (1, num_seeds, 2) ⇒ (H*W, num_seeds)
        eq = jnp.all(
            flat_coords[:, None, :] == seeds[None, :, :], axis=-1
        )  # shape (H*W, num_seeds)
        # argmax finds the first True per row
        indices = jnp.argmax(eq, axis=-1)
        # indices: (H*W,), reshape to grid
        arr2d = indices.reshape(arr.shape[0], arr.shape[1])
        return arr2d

    def create_weighted_palette(self, index_map: Array, data: Array):
        data_normed = (data - data.min()) / (data.max() - data.min())

        flat_index_map = index_map.ravel()
        flat_weights = data_normed.ravel()

        weights_sum = jnp.bincount(flat_index_map, weights=flat_weights)
        index_sum = jnp.bincount(flat_index_map)

        avg_voro_w = weights_sum / index_sum

        palette = jnp.stack(
            [avg_voro_w * 255, avg_voro_w * 255, avg_voro_w * 255], axis=1
        )
        palette = jnp.round(palette).astype(jnp.int32)

        return palette

    def create_lbg_palette(self, lower_mask, upper_mask):
        # Should return a palette with red for lower mask voronois
        # and green for upper mask voronois, showing what cells
        # are deleted and what are split
        size = lower_mask.shape[0]

        red = jnp.tile(jnp.array([252, 54, 5]), (size, 1))
        white = jnp.tile(jnp.array([252, 252, 252]), (size, 1))
        green = jnp.tile(jnp.array([54, 252, 5]), (size, 1))

        palette = jnp.where(lower_mask[:, None], red, white)
        palette = jnp.where(upper_mask[:, None], green, palette)
        return palette

    def create_random_palette(self):
        with jax.default_device(jax.devices("cpu")[0]):

            def set_colour(i):
                key = jax.random.PRNGKey(i)
                return (jax.random.uniform(key, shape=(3,)) * 255).astype(jnp.int32)

            return jax.vmap(set_colour)(jnp.arange(self.numseeds + 1))

    def get_colour_map(self, index_map: Array, palette: Array):
        return palette[index_map]


@staticmethod
@jax.jit
def lloyd_step(index_map: Array, data: Array, seeds: Array) -> Array:
    if data.shape[0] != index_map.shape[0] or data.shape[1] != index_map.shape[1]:
        print("Voronoi size does not match data size")
        return seeds
    # Normalize data for optional weighting
    data_normed = 1 - (data - data.min()) / (data.max() - data.min())

    # Build grid of coordinate indices
    grid = jnp.arange(data.shape[0])
    xx, yy = jnp.meshgrid(grid, grid)
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # shape (N, 2)

    flat_index_map = index_map.ravel()
    flat_weights = data_normed.ravel()

    # Weighted sum of x and y coordinates for each region
    offset_x = coords[:, 0] * flat_weights  # shape (N,)
    offset_y = coords[:, 1] * flat_weights
    sum_x = jnp.bincount(flat_index_map, weights=offset_x, length=len(seeds))
    sum_y = jnp.bincount(flat_index_map, weights=offset_y, length=len(seeds))
    weights_sum = jnp.bincount(flat_index_map, weights=flat_weights, length=len(seeds))

    lerp_t = 0.1
    # Avoid division by zero for empty regions
    float_seeds = jnp.stack(
        [
            jnp.where(weights_sum > 0, sum_y / weights_sum, seeds[:, 0]),
            jnp.where(weights_sum > 0, sum_x / weights_sum, seeds[:, 1]),
        ],
        axis=-1,
    )
    new_seeds = (1 - lerp_t) * seeds + lerp_t * float_seeds
    return new_seeds


def lbg_step(jfa_map: Array, data: Array, seeds: Array):
    if data.shape[0] != jfa_map.shape[0] or data.shape[1] != jfa_map.shape[1]:
        print("Voronoi size does not match data size")
        return seeds

    num_seeds = len(seeds)

    index_map = Voronoi.get_index_map(jfa_map, seeds)
    border_dist_transform = Voronoi.get_border_distance_transform(jfa_map)
    dist_transform = Voronoi.get_distance_transform(jfa_map, 0)
    _, unit_vectors = Voronoi.get_largest_extent(index_map, dist_transform, seeds)
    # _, circ_r = Voronoi.get_inscribing_circles(index_map, border_dist_transform, len(seeds))
    flat_index_map = index_map.ravel()
    flat_dist_transform = border_dist_transform.ravel()
    circ_r = jax.ops.segment_max(flat_dist_transform, flat_index_map, num_seeds)
    centroids = Voronoi.get_voro_centroids(index_map, num_seeds)
    split_coords = Voronoi.get_split(
        unit_vectors, circ_r / 2, centroids
    )  # TODO Split coords are slightly different to in main.py

    # Normalize data for optional weighting
    data_normed = (data - data.min()) / (data.max() - data.min())
    # Add a small value to data so that blank areas still create large voronoi cells
    data_normed = data_normed + 0.0001

    flat_index_map = index_map.ravel()
    flat_weights = data_normed.ravel()

    weights_sum = jnp.bincount(flat_index_map, weights=flat_weights, length=num_seeds)

    lower_mask = weights_sum < LOWER_THRESHOLD
    upper_mask = weights_sum > UPPER_THRESHOLD

    remove_mask = lower_mask | upper_mask  # Logical OR
    new_seeds = jnp.concat(
        [centroids[~remove_mask], split_coords[upper_mask].reshape(-1, 2)]
    )

    return new_seeds, lower_mask, upper_mask


if __name__ == "__main__":
    size = 2000
    key = jax.random.PRNGKey(size)
    with jax.default_device(jax.devices("cpu")[0]):
        seeds = (jax.random.uniform(key, shape=(NUM_SEEDS, 2)) * size).astype(jnp.int32)

    voro = Voronoi(size=size, seeds=seeds)
    jfa_map = voro.jfa()
    index_map = voro.get_index_map(jfa_map, seeds)
    colour_map = voro.get_colour_map(index_map)
