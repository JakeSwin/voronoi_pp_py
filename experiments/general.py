import jax.numpy as jnp
import jax

from src.voronoi import Voronoi

texture_size = 50
key = jax.random.PRNGKey(texture_size)
seeds = jax.random.uniform(key, shape=(5, 2)) * texture_size
num_seeds = seeds.shape[0]

vr = Voronoi(texture_size, seeds)

jfa_map = vr.jfa()
index_map = vr.get_index_map(jfa_map, seeds)
