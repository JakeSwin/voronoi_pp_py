import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr

from jax import lax
from PIL import Image

from src.voronoi import Voronoi, lbg_step
from src.gps.numpyro_gp import GP
from src.util import normalize_min_max


def main():
    rr.init("voronoi_jump_flooding", spawn=True)
    # rr.save(f"/home/swin/datasets/rerun/lower_{LOWER_THRESHOLD}_upper_{UPPER_THRESHOLD}.rrd")

    im = Image.open("./images/first000_gt.png")
    jnp_im = jnp.array(im)

    width = jnp_im.shape[0]
    height = jnp_im.shape[1]
    crop_size = 150
    texture_size = 500
    half_crop = crop_size // 2
    num_samples = 1000

    key = jax.random.PRNGKey(123)
    x = jax.random.randint(
        key=key, shape=(num_samples,), minval=crop_size, maxval=width - crop_size + 1
    )
    y = jax.random.randint(
        key=key, shape=(num_samples,), minval=crop_size, maxval=height - crop_size + 1
    )
    samples = jnp.column_stack([x, y])

    def get_weed_chance(coord):
        x = coord[0]
        y = coord[1]
        crop_shape = (crop_size, crop_size, 3)
        start_indices = (x - half_crop, y - half_crop, 0)
        crop = lax.dynamic_slice(jnp_im, start_indices, crop_shape)
        avg_pool = jnp.count_nonzero(crop.ravel()) / crop_size**2
        return avg_pool

    weed_chance = jax.vmap(get_weed_chance)(samples)

    # gp = GP(texture_size, width, height)
    # gp.add_samples(samples, weed_chance)
    # gp.optimise()
    # gp_map = gp.predict_map()
    gp = GP(texture_size, width, height)
    gp.fit(samples, weed_chance)
    gp_map = gp.predict_map(samples, weed_chance)
    clamped_gp_map = jnp.where(gp_map[0] < 0, 0.0, gp_map[0])
    normed_mean_map, _, _ = normalize_min_max(clamped_gp_map)
    normed_mean_map = normed_mean_map + 0.0001
    normed_mean_map = normed_mean_map * 255
    print(gp_map[0].shape)

    key = jax.random.PRNGKey(texture_size)
    with jax.default_device(jax.devices("cpu")[0]):
        new_seeds = jax.random.uniform(key, shape=(500, 2)) * texture_size
    num_samples = new_seeds.shape[0]

    vr = Voronoi(texture_size, new_seeds)
    jfa_map = vr.jfa()
    border_dist_transform = vr.get_border_distance_transform(jfa_map)
    dist_transform = vr.get_distance_transform(jfa_map, 0)
    index_map = vr.get_index_map(jfa_map, new_seeds)
    _, unit_vectors = vr.get_largest_extent(index_map, dist_transform, new_seeds)
    circ_points, circ_r = vr.get_inscribing_circles(
        index_map, border_dist_transform, num_samples
    )
    palette = vr.create_weighted_palette(index_map, clamped_gp_map)
    colour_map = vr.get_colour_map(index_map, palette)
    centroids = vr.get_voro_centroids(index_map, num_samples)
    split_coords = vr.get_split(unit_vectors, circ_r / 2, centroids)

    rr.log("GP/Image", rr.Image(normed_mean_map))
    rr.log("GP/Samples", rr.Points2D(jnp.flip(new_seeds, axis=1)))
    rr.log("GP/Voronoi", rr.Image(colour_map))
    rr.log("GP/Voronoi/Distance", rr.Image(border_dist_transform))
    rr.log("GP/Voronoi/Centroids", rr.Points2D(centroids))
    rr.log("GP/Voronoi/InscribingCircles", rr.Points2D(circ_points, radii=circ_r))
    rr.log("GP/Voronoi/Splits", rr.LineStrips2D(np.asarray(split_coords)))

    count = 0
    while True:
        # new_seeds = lloyd_step(index_map, gp_map[0], new_seeds)
        prev_seeds = new_seeds
        new_seeds, lower_mask, upper_mask = lbg_step(jfa_map, clamped_gp_map, new_seeds)
        total_changed_percent = (lower_mask.sum() + upper_mask.sum()) / num_samples
        prev_num_samples = num_samples
        num_samples = new_seeds.shape[0]
        print(f"{count}): {new_seeds.shape}")
        vr.setup(texture_size, new_seeds)
        jfa_map = vr.jfa()
        border_dist_transform = vr.get_border_distance_transform(jfa_map)
        dist_transform = vr.get_distance_transform(jfa_map, 0)
        index_map = vr.get_index_map(jfa_map, new_seeds)
        _, unit_vectors = vr.get_largest_extent(index_map, dist_transform, new_seeds)
        circ_points, circ_r = vr.get_inscribing_circles(
            index_map, border_dist_transform, num_samples
        )
        weighted_palette = vr.create_weighted_palette(index_map, clamped_gp_map)
        lbg_palette = vr.create_lbg_palette(lower_mask, upper_mask)
        colour_map = vr.get_colour_map(index_map, weighted_palette)
        colour_map_splits = vr.get_colour_map(index_map, lbg_palette)
        centroids = vr.get_voro_centroids(index_map, num_samples)
        split_coords = vr.get_split(unit_vectors, circ_r / 2, centroids)

        rr.log("GP/Image", rr.Image(normed_mean_map))
        rr.log("GP/Samples", rr.Points2D(jnp.flip(new_seeds, axis=1)))
        rr.log("GP/Voronoi", rr.Image(colour_map))
        rr.log("GP/Voronoi/SplitMap", rr.Image(colour_map_splits))
        rr.log("GP/Voronoi/Distance", rr.Image(border_dist_transform))
        rr.log("GP/Voronoi/Centroids", rr.Points2D(centroids))
        rr.log("GP/Voronoi/InscribingCircles", rr.Points2D(circ_points, radii=circ_r))
        rr.log("GP/Voronoi/Splits", rr.LineStrips2D(np.asarray(split_coords)))

        print(f"Total Changed Percent: {total_changed_percent}")
        if total_changed_percent < 0.25:
            break
        else:
            count += 1


if __name__ == "__main__":
    main()
