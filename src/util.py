import jax.numpy as jnp

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
