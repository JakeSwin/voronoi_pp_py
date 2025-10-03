import jax
import random
import jax.numpy as jnp
import rerun as rr

from jax import lax
from PIL import Image

from src.voronoi import Voronoi
from src.gp import GP
from src.util import normalize, normalize_coords
from src.sample import weighted_sample_elimination

def main():
    rr.init("voronoi_jump_flooding", spawn=True)

    im = Image.open("./images/first000_gt.png")
    jnp_im = jnp.array(im)

    width = jnp_im.shape[0]
    height = jnp_im.shape[1]
    crop_size = 150
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

    gp = GP(500, width, height)
    gp.add_samples(samples, weed_chance)
    opt_post = gp.optimise_posterior_sparse()
    gp_map = gp.predict_map(opt_post)
    print(gp_map[0].shape)

    # rr.log(
    #     "Voronoi/Points",
    #     rr.Points2D(samples, radii=weed_chance*10)
    # )

    # gp_samples = weighted_sample_elimination(gp_map[0])
    num_gp_samples = 500
    gp_samples = []
    while len(gp_samples) < num_gp_samples:
        x = random.randint(0, gp_map[0].shape[0])
        y = random.randint(0, gp_map[0].shape[1])
        if gp_map[0][x, y] > random.random():
            gp_samples.append([x, y])

    vr = Voronoi(500, jnp.array(gp_samples))
    jfa_map = vr.jfa()
    index_map = vr.get_index_map(jfa_map, jnp.array(gp_samples))
    colour_map = vr.get_colour_map(index_map)

    rr.log("GP/Image", rr.Image(gp_map[0]))
    rr.log("GP/Samples", rr.Points2D(gp_samples))
    rr.log("GP/Voronoi", rr.Image(colour_map))


if __name__ == "__main__":
    main()
