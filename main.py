import jax
import random
import jax.numpy as jnp
import numpy as np
import rerun as rr

from jax import lax
from PIL import Image

from src.voronoi import Voronoi, lloyd_step, lbg_step
from src.gp import GP
from src.util import normalize, normalize_coords, jaccard_similarity
from src.sample import weighted_sample_elimination
from src.constants import LOWER_THRESHOLD, UPPER_THRESHOLD

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
    num_samples = 2000

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

    gp = GP(texture_size, width, height)
    gp.add_samples(samples, weed_chance)
    opt_post = gp.optimise_posterior_sparse()
    gp_map = gp.predict_map(opt_post)
    print(gp_map[0].shape)

    # rr.log(
    #     "Voronoi/Points",
    #     rr.Points2D(samples, radii=weed_chance*10)
    # )

    # gp_samples = weighted_sample_elimination(gp_map[0])
    # num_gp_samples = 500
    # gp_samples = []
    # while len(gp_samples) < num_gp_samples:
    #     x = random.randint(0, gp_map[0].shape[0])
    #     y = random.randint(0, gp_map[0].shape[1])
    #     if gp_map[0][x, y] > random.random():
    #         gp_samples.append([x, y])

    # new_seeds = jnp.array(gp_samples)
    # new_seeds = jnp.array([[texture_size/2, texture_size/2]])
    key = jax.random.PRNGKey(texture_size)
    with jax.default_device(jax.devices("cpu")[0]):
        new_seeds = (jax.random.uniform(key, shape=(500, 2)) * texture_size)
    num_samples = new_seeds.shape[0]

    vr = Voronoi(texture_size, new_seeds)
    jfa_map = vr.jfa()
    border_dist_transform = vr.get_border_distance_transform(jfa_map)
    dist_transform = vr.get_distance_transform(jfa_map, 0)
    index_map = vr.get_index_map(jfa_map, new_seeds)
    _, unit_vectors = vr.get_largest_extent(index_map, dist_transform, new_seeds)
    circ_points, circ_r = vr.get_inscribing_circles(index_map, border_dist_transform, num_samples)
    palette = vr.create_weighted_palette(index_map, gp_map[0])
    colour_map = vr.get_colour_map(index_map, palette)
    centroids = vr.get_voro_centroids(index_map, num_samples)
    split_coords = vr.get_split(unit_vectors, circ_r/2, centroids)

    rr.log("GP/Image", rr.Image(gp_map[0]))
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
        new_seeds, total_changed_percent = lbg_step(jfa_map, gp_map[0], new_seeds)
        prev_num_samples = num_samples
        num_samples = new_seeds.shape[0]
        print(f"{count}): {new_seeds.shape}")
        vr.setup(texture_size, new_seeds)
        jfa_map = vr.jfa()
        border_dist_transform = vr.get_border_distance_transform(jfa_map)
        dist_transform = vr.get_distance_transform(jfa_map, 0)
        index_map = vr.get_index_map(jfa_map, new_seeds)
        _, unit_vectors = vr.get_largest_extent(index_map, dist_transform, new_seeds)
        circ_points, circ_r = vr.get_inscribing_circles(index_map, border_dist_transform, num_samples)
        palette = vr.create_weighted_palette(index_map, gp_map[0])
        colour_map = vr.get_colour_map(index_map, palette)
        centroids = vr.get_voro_centroids(index_map, num_samples)
        split_coords = vr.get_split(unit_vectors, circ_r/2, centroids)

        rr.log("GP/Image", rr.Image(gp_map[0]))
        rr.log("GP/Samples", rr.Points2D(jnp.flip(new_seeds, axis=1)))
        rr.log("GP/Voronoi", rr.Image(colour_map))
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
