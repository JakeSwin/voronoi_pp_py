import jax.numpy as jnp

from src.constants import MAX_CELLS

def normalize(array):
    mean = jnp.mean(array)
    std = jnp.std(array)
    normed = (array - mean) / std
    return normed, mean, std

def unnormalize(normed_array, mean, std):
    original = normed_array * std + mean
    return original

def normalize_coords(coords, mins=None, maxs=None):
    if mins is None or maxs is None:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
    normed = (coords - mins) / (maxs - mins)
    return normed, mins, maxs

def normalize_min_max(coords, min=None, max=None):
    if min is None or max is None:
        min = coords.min()
        max = coords.max()
    normed = (coords - min) / (max - min)
    return normed, min, max

def prepare_seeds(raw_seeds):
    num_seeds = raw_seeds.shape[0]
    seeds_padded = jnp.pad(raw_seeds, ((0, MAX_CELLS - num_seeds), (0, 0)))
    mask = jnp.arange(MAX_CELLS) < num_seeds
    return seeds_padded, mask
