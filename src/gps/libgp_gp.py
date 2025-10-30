# Not working very well at all

import libgp
import numpy as np

class GP:
    def __init__(self, out_size: int, input_width: float, input_height: float):
        self.out_size = out_size
        self.input_width = input_width
        self.input_height = input_height

        self.length_scale = np.log(0.08)
        self.signal_var = np.log(0.15)
        self.noise_var = np.log(0.001)

        self.gp = libgp.GaussianProcess(2, "CovSum(CovSEiso, CovNoise)")

        params = np.array([self.length_scale, self.signal_var, self.noise_var])
        self.gp.set_loghyper(params)

    def normalize_coords(self, coords):
        coords = np.asarray(coords)
        x_norm = coords[..., 0] / self.input_width
        y_norm = coords[..., 1] / self.input_height
        return np.stack([x_norm, y_norm], axis=-1)

    def add_samples(self, coords, data):  # coords: (N,2), data: (N,)
        coords_norm = self.normalize_coords(coords)
        self.gp.add_patterns(coords_norm, data)

    def add_sample(self, coord, data):  # coord: (2,), data: float
        coord_norm = self.normalize_coords(np.array(coord).reshape(1, 2))[0]
        self.gp.add_pattern(coord_norm, data)

    def optimise(self):
        optimizer = libgp.OptimizerRProp(1e-6, 0.5, 1e-5, 20, 0.3, 1.3)
        optimizer.maximize(self.gp, 25, True)

    def predict_map(self):
        x_grid = np.linspace(0, self.input_width, self.out_size)
        y_grid = np.linspace(0, self.input_height, self.out_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        X_test = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        X_test_norm = self.normalize_coords(X_test)

        batch_size = 2000
        means = []
        variances = []
        for i in range(0, X_test_norm.shape[0], batch_size):
            X_batch = X_test_norm[i : i + batch_size]
            result = self.gp.predict_with_variance(X_batch)
            means.append(result[0])
            variances.append(result[1])

        mean_map = np.array(means).reshape((self.out_size, self.out_size))
        cov_map = np.array(variances).reshape((self.out_size, self.out_size))

        return mean_map, cov_map
