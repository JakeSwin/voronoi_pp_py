import jax.numpy as jnp


class UAV:
    def __init__(self, start_pos):
        self.pos = jnp.array(start_pos)
        self.path = None

    def move(self, next_pos):
        if self.path == None:
            self.path = self.pos[None, :]
        else:
            self.path = jnp.concatenate((self.path, self.pos[None, :]))
        self.pos = next_pos
