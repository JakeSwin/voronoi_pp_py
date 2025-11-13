import jax
import jax.numpy as jnp
import rerun as rr

from PIL import Image
from jax import lax

from src.util import normalize_min_max
from src.gps.numpyro_gp import GP
from src.planner import Planner
from src.voronoi import Voronoi, lbg_step
from src.uav import UAV

crop_size = 150
texture_size = 500
half_crop = crop_size // 2
num_samples = 1000
num_starting_seeds = 500

def get_maps(gp, X, y):
    mean_map, cov_map = gp.predict_map(X, y)
    mean_map = jnp.where(mean_map < 0, 0.0, mean_map)
    mean_map, _, _ = normalize_min_max(mean_map)
    mean_map = mean_map + 0.0001
    return mean_map, cov_map

prev_changed_percent = 0

def should_lbg_step(lower_mask, upper_mask, num_seeds):
    global prev_changed_percent
    changed_percent = (lower_mask.sum() + upper_mask.sum()) / num_seeds
    rate_of_change = jnp.abs(changed_percent - prev_changed_percent)
    print(f"Changed Percent: {changed_percent}, Rate of Change: {rate_of_change}")
    prev_changed_percent = changed_percent
    return (changed_percent > 0.25 and rate_of_change > 0.05)

def fit_voronoi_lbg(seeds, num_seeds, voronoi, mean_map):
    count = 0
    jfa_map = voronoi.jfa()
    while True:
        seeds, lower_mask, upper_mask = lbg_step(jfa_map, mean_map, seeds)
        print(f"{count}): {seeds.shape}")
        prev_num_seeds = num_seeds
        num_seeds = seeds.shape[0]
        voronoi.setup(texture_size, seeds)
        jfa_map = voronoi.jfa()
        if should_lbg_step(lower_mask, upper_mask, prev_num_seeds):
            count += 1
            continue
        else:
            break
    return seeds, num_seeds, jfa_map

if __name__ == "__main__":
    rr.init("voro_pp_exploration", spawn=True)

    im = Image.open("./images/first000_gt.png")
    jnp_im = jnp.array(im)

    width = jnp_im.shape[0]
    height = jnp_im.shape[1]

    key = jax.random.PRNGKey(123)
    x = jax.random.randint(
        key=key, shape=(num_samples,), minval=crop_size, maxval=width - crop_size + 1
    )
    y = jax.random.randint(
        key=key, shape=(num_samples,), minval=crop_size, maxval=height - crop_size + 1
    )
    samples = jnp.column_stack([x, y])

    def get_weed_chance(coord):
        x = jnp.clip(jnp.round(coord[0]), 0, width).astype(jnp.int32)
        y = jnp.clip(jnp.round(coord[1]), 0, height).astype(jnp.int32)
        crop_shape = (crop_size, crop_size, 3)
        start_indices = (x - half_crop, y - half_crop, 0)
        crop = lax.dynamic_slice(jnp_im, start_indices, crop_shape)
        avg_pool = jnp.count_nonzero(crop.ravel()) / crop_size**2
        return avg_pool

    weed_chance = jax.vmap(get_weed_chance)(samples)

    normed_samples, _, _ = normalize_min_max(samples, crop_size, width - crop_size + 1)

    gp = GP(texture_size, 1.0, height / width)
    # gp.fit(normed_samples, weed_chance)
    gp.fit(jnp.empty((0, 2)), jnp.empty((0,)))

    # mean_map, cov_map = get_maps(gp, normed_samples, weed_chance)

    key = jax.random.PRNGKey(texture_size)
    seeds = jax.random.uniform(key, shape=(num_starting_seeds, 2)) * texture_size
    num_seeds = seeds.shape[0]

    vr = Voronoi(texture_size, seeds)

    uav_sample_func = lambda coord: get_weed_chance((coord/texture_size)*width)

    uav = UAV([texture_size/2, texture_size/2], uav_sample_func)
    planner = Planner()

    while True:
        path = uav.get_full_path()
        normed_coords, _, _ = normalize_min_max(((path/texture_size)*width), 0, width)
        mean_map, _ = get_maps(gp, normed_coords, uav.samples)
        rr.log("GP/Image", rr.Image(mean_map * 255))

        seeds, num_seeds, jfa_map = fit_voronoi_lbg(seeds, num_seeds, vr, mean_map)
        index_map = vr.get_index_map(jfa_map, seeds)
        centroids = vr.get_voro_centroids(index_map, num_seeds)
        rr.log("Voronoi/Centroids", rr.Points2D(centroids))

        planner.step(uav, index_map, centroids)
