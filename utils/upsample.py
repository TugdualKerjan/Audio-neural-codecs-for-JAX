import equinox as eqx
import jax

class Upsample1d(eqx.Module):
    scale_factor: int
    mode: str

    def __init__(self, scale_factor, mode):
        self.scale_factor = scale_factor
        self.mode = mode

    @eqx.filter_jit
    def __call__(self, x):
        new_height = x.shape[1] * self.scale_factor
        return jax.image.resize(x, (x.shape[0], new_height), method=self.mode)