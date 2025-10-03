import jax
import jax.numpy as jnp
import time

from jaxtyping import Array

from src.constants import NUM_SEEDS

offset_vectors = jnp.array([[-1, -1], [-1, 0], [-1, 1],
                     [0, -1],  [0, 0],  [0, 1],
                     [1, -1],  [1, 0],  [1, 1]])  # For jump, scale by offset

class Voronoi:
    def __init__(self, size: int, seeds: Array):
        self.setup(size, seeds)

    def update(self, size: int, seeds: Array):
        self.setup(size, seeds)

    def setup(self, size: int, seeds: Array):
        self.size = size
        self.numseeds = seeds.shape[0]

        with jax.default_device(jax.devices("cpu")[0]):
            arr = jnp.zeros((size, size, 2))
            self.seed_map = arr.at[seeds[:, 0], seeds[:, 1]].set(seeds)

            max_exp = int(jnp.floor(jnp.log2(size // 2)))
            # JFA+2 variant for more accurate border
            self.jfa_offsets = [(size // 2) // (2 ** i) for i in range(max_exp + 1)] + [1, 1]

            # Generate Colour Pallete
            self.palette = self._create_palette()

    @staticmethod
    @jax.jit
    def _make_seeded_arr(size: int, seeds: Array) -> Array:
        arr = jnp.zeros((size, size, 2))
        return arr.at[seeds[:, 0], seeds[:, 1]].set(seeds)

    @staticmethod
    @jax.jit
    def _step_fn(arr: Array, offset: int):
        size = arr.shape[0]

        def step(i, j):
            pos = jnp.array([i, j])
            shifts = offset_vectors * offset
            candidates = pos + shifts
            candidates = jnp.clip(candidates, 0, size - 1)
            info = arr[candidates[:,0], candidates[:,1]]
            dists = jnp.linalg.norm(info - pos, axis=-1)
            argmin = jnp.argmin(dists)
            return info[argmin]

        vmapped = jax.vmap(jax.vmap(step, in_axes=(None, 0)), in_axes=(0, None))
        grid = jnp.arange(size)
        return vmapped(grid, grid)

    def jfa(self):
        arr = self.seed_map
        for offset in self.jfa_offsets:
            arr = self._step_fn(arr, offset)
        return arr

    def get_index_map(self, jfa_map: Array, seeds: Array):
        return self._index_map(jfa_map, seeds)

    @staticmethod
    @jax.jit
    def _index_map(arr: Array, seeds: Array):
        flat_coords = arr.reshape(-1, arr.shape[-1]) # shape (H*W, 2)
        seeds = jnp.array(seeds)
        # Get each pixel's index in seeds (by matching coordinates for each pixel)
        # Broadcasting: (H*W, 1, 2) == (1, num_seeds, 2) ⇒ (H*W, num_seeds)
        eq = jnp.all(flat_coords[:, None, :] == seeds[None, :, :], axis=-1) # shape (H*W, num_seeds)
        # argmax finds the first True per row
        indices = jnp.argmax(eq, axis=-1)
        # indices: (H*W,), reshape to grid
        arr2d = indices.reshape(arr.shape[0], arr.shape[1])
        return arr2d

    def _create_palette(self):
        with jax.default_device(jax.devices("cpu")[0]):
            def set_colour(i):
                key = jax.random.PRNGKey(i)
                return (jax.random.uniform(key, shape=(3,)) * 255).astype(jnp.int32)

            return jax.vmap(set_colour)(jnp.arange(self.numseeds + 1))

    def get_colour_map(self, index_map: Array):
        return self.palette[index_map]

def lloyd_step(voro: Voronoi, data: Array, seeds: Array) -> Array:
    if data.shape[0] != voro.size or data.shape[1] != voro.size:
        print("Voronoi size does not match data size")
        return seeds
    new_seeds = seeds
    jfa_map = voro.jfa()
    index_map = voro.get_index_map(jfa_map, seeds)
    return new_seeds

if __name__ == "__main__":
    size = 2000
    key = jax.random.PRNGKey(size)
    with jax.default_device(jax.devices("cpu")[0]):
        seeds = (jax.random.uniform(key, shape=(NUM_SEEDS, 2)) * size).astype(jnp.int32)

    voro = Voronoi(size=size, seeds=seeds)
    jfa_map = voro.jfa()
    index_map = voro.get_index_map(jfa_map, seeds)
    colour_map = voro.get_colour_map(index_map)
