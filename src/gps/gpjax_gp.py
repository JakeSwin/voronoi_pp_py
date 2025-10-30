import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import optax as ox

from jaxtyping import Array
from gpjax.parameters import Parameter, PositiveReal, Real
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

from src.util import normalize, unnormalize, normalize_coords

# Enables Float64 for more stable matrix inversions
from jax import config

config.update("jax_enable_x64", True)

key = jr.key(123)

class GP:
    def __init__(self, out_size: int, input_width: float, input_height: float):
        self.out_size = out_size
        self.input_width = input_width
        self.input_height = input_height

        # Define kernel, mean and prior
        self.kernel = gpx.kernels.RBF(
            n_dims=2,
            lengthscale=PositiveReal(jnp.array([0.01, 0.01])),
            variance=PositiveReal(jnp.array([1.0]))
        ) + gpx.kernels.Matern12(
            n_dims=2,
            lengthscale=PositiveReal(jnp.array([0.01, 0.01])),
            variance=PositiveReal(jnp.array([1.0])))

        # Mean function should be the mean weed count accross the whole field?
        self.meanf = gpx.mean_functions.Constant(PositiveReal(jnp.array([1.0])))
        self.prior = gpx.gps.Prior(mean_function=self.meanf, kernel=self.kernel)

        # Initialize sample attributes
        self.coords = None
        self.data = None

    def add_samples(self, coords: Array, data: Array):
        if self.coords is None or self.data is None:
            self.coords = coords.astype(jnp.float64)
            self.data = data[:, None]
        else:
            self.coords = jnp.append(self.coords, coords, axis=0)
            self.data = jnp.append(self.data, data)[:, None]
        normalized_data, self.data_mean, self.data_std = normalize(self.data)
        normalized_coords, self.coords_min, self.coords_max = normalize_coords(self.coords)
        self.dataset = gpx.Dataset(X=normalized_coords, y=normalized_data)

    def optimise_posterior_sparse(self):
        if self.coords is None or self.data is None:
            print("Need to add training data before optimisation can occur")
            return

        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=self.coords.shape[0], obs_stddev=PositiveReal(jnp.array([0.1]))
        )
        posterior = self.prior * likelihood

        n_inducing_per_dim = 10

        kmeans = KMeans(n_clusters=n_inducing_per_dim).fit(
            normalize_coords(self.coords, self.coords_min, self.coords_max)[0],
            sample_weight=self.data.flatten()
        )

        # coords_scaled = StandardScaler().fit_transform(self.coords)
        # db = DBSCAN(eps=0.2, min_samples=30).fit(coords_scaled)
        # core_samples = self.coords[db.core_sample_indices_]

        q = gpx.variational_families.CollapsedVariationalGaussian(
            posterior=posterior, inducing_inputs=kmeans.cluster_centers_
        )

        opt_posterior, _ = gpx.fit(
            model=q,
            objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
            train_data=self.dataset,
            optim=ox.adamw(learning_rate=1e-2),
            num_iters=5000,
            key=key,
            trainable=Parameter,
        )

        return opt_posterior

    def optimise_posterior(self):
        if self.coords is None or self.data is None:
            print("Need to add training data before optimisation can occur")
            return

        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=self.coords.shape[0], obs_stddev=PositiveReal(jnp.array([0.1]))
        )
        posterior = self.prior * likelihood

        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=self.dataset,
            optim=ox.adamw(learning_rate=1e-2),
            num_iters=500,
            key=key,
            trainable=Parameter,
        )

        return opt_posterior

    def predict_map(self, posterior):
        if self.coords is None or self.data is None:
            print("No training data found, outputting empty map")
            zero_map = jnp.zeros((self.out_size, self.out_size), dtype=jnp.float32)
            return zero_map, zero_map

        x_grid = jnp.linspace(0, self.input_width, self.out_size)
        y_grid = jnp.linspace(0, self.input_height, self.out_size)
        xx, yy = jnp.meshgrid(x_grid, y_grid)
        X_test = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
        X_test, _, _ = normalize_coords(X_test, self.coords_min, self.coords_max)

        batch_size = 2000
        means = []
        variances = []
        for i in range(0, X_test.shape[0], batch_size):
            X_batch = X_test[i : i + batch_size]
            latent = posterior.predict(X_batch, train_data=self.dataset)
            means.append(latent.mean)
            variances.append(latent.stddev())
        predicted_mean = unnormalize(
            jnp.concatenate(means), self.data_mean, self.data_std
        ).reshape((self.out_size, self.out_size))
        predicted_std = jnp.concatenate(variances).reshape(
            (self.out_size, self.out_size)
        )

        return predicted_mean, predicted_std
