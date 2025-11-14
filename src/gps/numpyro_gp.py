import jax
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS

from src.util import normalize_min_max


def rbf_kernel(X1, X2, var, length):
    dists = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    return var * jnp.exp(-0.5 * dists / (length**2))


def matern32_kernel(X1, X2, var_m, length_m):
    dists = jnp.sqrt(jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
    sqrt3 = jnp.sqrt(3.0)
    scaled = sqrt3 * dists / length_m
    return var_m * (1.0 + scaled) * jnp.exp(-scaled)


def combined_kernel(
    X1, X2, var, length, var_m, length_m, noise=0, jitter=1e-6, include_noise=True
):
    k_rbf = rbf_kernel(X1, X2, var, length)
    k_matern = matern32_kernel(X1, X2, var_m, length_m)
    k = k_rbf + k_matern
    if include_noise:
        k = k + (noise + jitter) * jnp.eye(X1.shape[0])
    return k


kernel_jit = jax.jit(combined_kernel, static_argnames=["include_noise"])


def gp_model(X, Y=None, jitter=1e-6):
    var = numpyro.sample("var", dist.LogNormal(0.0, 0.5))
    var_m = numpyro.sample("var_m", dist.LogNormal(0.0, 0.5))
    # length = numpyro.sample("length", dist.LogNormal(0.0, 1.0))
    # noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
    # Favor larger smoothness
    length = numpyro.sample("length", dist.LogNormal(jnp.log(0.05), 0.2))
    length_m = numpyro.sample("length_m", dist.LogNormal(jnp.log(0.05), 0.2))
    # Prevent noise from shrinking to zero
    noise = numpyro.sample("noise", dist.LogNormal(-3.0, 0.3))
    k = kernel_jit(X, X, var, length, var_m, length_m, noise, jitter)
    numpyro.sample("obs", dist.MultivariateNormal(jnp.zeros(X.shape[0]), k), obs=Y)


def fit_gp(X_train, Y_train, num_warmup=500, num_samples=1000):
    rng_key = jax.random.PRNGKey(0)
    mcmc = MCMC(NUTS(gp_model), num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X_train, Y_train)
    return mcmc.get_samples()


def predict_gp(
    X_train, Y_train, X_test, var, length, var_m, length_m, noise=0, jitter=1e-6
):
    K = kernel_jit(X_train, X_train, var, length, var_m, length_m, noise, jitter)
    K_s = kernel_jit(
        X_train, X_test, var, length, var_m, length_m, 0, include_noise=False
    )
    K_ss = kernel_jit(X_test, X_test, var, length, var_m, length_m, 0)

    K_inv = jnp.linalg.inv(K)
    mu_s = jnp.dot(K_s.T, jnp.dot(K_inv, Y_train))
    cov_s = K_ss - jnp.dot(K_s.T, jnp.dot(K_inv, K_s))
    return mu_s, cov_s


predict_gp_jit = jax.jit(predict_gp)


class GP:
    def __init__(self, out_size: int, input_width: float, input_height: float):
        self.out_size = out_size
        self.input_width = input_width
        self.input_height = input_height

    def fit(self, X_train, Y_train, num_warmup=500, num_samples=1000):
        self.samples = fit_gp(X_train, Y_train, num_warmup, num_samples)
        params = {
            key: self.samples[key].mean()
            for key in ["var", "length", "noise", "var_m", "length_m"]
        }
        print(
            "Fitted variance:",
            params["var"],
            "Fitted length:",
            params["length"],
            "Fitted variance Matern:",
            params["var_m"],
            "Fitted length Matern:",
            params["length_m"],
            "Fitted noise:",
            params["noise"],
        )

    def predict(self, X_train, Y_train, X_test):
        # Use mean or sample of hyperparameters to call predict_gp
        params = {
            key: self.samples[key].mean()
            for key in ["var", "length", "noise", "var_m", "length_m"]
        }
        if X_train.shape[0] == 0:
            mu_s = jnp.zeros(X_test.shape[0])
            cov_s = kernel_jit(X_test, X_test, **params)
            return mu_s, cov_s
        else:
            return predict_gp_jit(X_train, Y_train, X_test, **params)

    def predict_map(self, X_train, Y_train):
        x_grid = jnp.linspace(0, self.input_width, self.out_size)
        y_grid = jnp.linspace(0, self.input_height, self.out_size)
        xx, yy = jnp.meshgrid(x_grid, y_grid)
        X_test = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

        batch_size = 2000
        means = []
        variances = []
        for i in range(0, X_test.shape[0], batch_size):
            X_batch = X_test[i : i + batch_size]
            result = self.predict(X_train, Y_train, X_batch)
            means.append(result[0])
            variances.append(jnp.diag(result[1]))

        mean_map = jnp.array(means).reshape((self.out_size, self.out_size))
        cov_map = jnp.array(variances).reshape((self.out_size, self.out_size))

        return mean_map, cov_map
