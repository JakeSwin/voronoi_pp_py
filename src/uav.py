import jax.numpy as jnp

import rerun as rr

class UAV:
    # Assume sample_func is a function that takes a single argument with a coordinate [x, y]
    # and returns the sample probability at that point
    def __init__(self, start_pos, sample_func):
        self.pos = jnp.array(start_pos)
        self.sample_func = sample_func

        sample = self.sample_func(self.pos)
        self.samples = jnp.atleast_1d(sample)

        self.path = None

    def move(self, next_pos):
        if self.path == None:
            self.path = self.pos[None, :]
        else:
            self.path = jnp.concatenate((self.path, self.pos[None, :]))
        self.pos = jnp.array(next_pos)
        self.sample()

    def sample(self):
        new_sample = self.sample_func(self.pos)
        new_sample = jnp.atleast_1d(new_sample)
        self.samples = jnp.concatenate((self.samples, new_sample))

    def get_full_path(self):
        if self.path != None:
            return jnp.concatenate((self.path, self.pos[None, :]))
        else:
            return self.pos[None, :]

    def log(self):
        rr.log("UAV/CurrentPos", rr.Points2D(self.pos.tolist()))
        rr.log("UAV/CurrentPos", rr.AnyValues(sample_value=self.samples[-1].tolist()))
        if self.path != None:
            full_path_pos = jnp.concatenate((self.path, self.pos[None, :]))
            windows = jnp.stack([full_path_pos[:-1], full_path_pos[1:]], axis=0)
            full_path_lines = jnp.transpose(windows, (1, 0, 2))

            rr.log("UAV/Path", rr.LineStrips2D(full_path_lines.tolist()))
            rr.log("UAV/PreviousPos", rr.Points2D(self.path.tolist()))
            rr.log("UAV/PreviousPos", rr.AnyValues(sample_value=self.samples[:-1].tolist()))
