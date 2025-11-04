import jax
import jax.numpy as jnp
import rerun as rr

from jax import lax
from PIL import Image

from src.gps.numpyro_gp import GP
from src.util import normalize_min_max


def get_normed_maps(gp, X, y):
    mean_map, cov_map = gp.predict_map(X, y)
    mean_map = jnp.where(mean_map < 0, 0.0, mean_map)
    mean_map, _, _ = normalize_min_max(mean_map)
    mean_map = mean_map * 255
    return mean_map, cov_map


if __name__ == "__main__":
    rr.init("online_gp", spawn=True)
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

    gt_gp = GP(texture_size, width, height)
    gt_gp.fit(samples, weed_chance)
    mean_map, _ = get_normed_maps(gt_gp, samples, weed_chance)

    rr.log("GP/GroundTruth", rr.Image(mean_map))

    # Setup lawnmower path
    x_points = jnp.linspace(crop_size, width - crop_size + 1, 15)
    y_points = jnp.linspace(crop_size, height - crop_size + 1, 15)
    xx, yy = jnp.meshgrid(x_points, y_points)

    reverse_mask = jnp.arange(x_points.shape[0]) % 2 == 1
    reverse_mask = reverse_mask[:, None]

    xx_flipped = jnp.where(reverse_mask, xx[:, ::-1], xx)
    coords = jnp.stack([xx_flipped.ravel(), yy.ravel()], axis=-1)
    gt_values = gt_gp.predict(samples, weed_chance, coords)[0]

    test_gp = GP(texture_size, width, height)
    test_gp.fit(jnp.empty((0, 2)), jnp.empty((0,)))

    for i in range(1, coords.shape[0]):
        mean_map, _ = get_normed_maps(test_gp, coords[:i], gt_values[:i])
        rr.log("GP/TestMap", rr.Image(mean_map))
        if i % 50 == 0:
            test_gp.fit(coords[:i], gt_values[:i])
