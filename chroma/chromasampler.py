# Copyright © 2024 Apple Inc.

from functools import lru_cache
import mlx.core as mx
from typing import Callable

class ChromaSampler:
    def __init__(self, name: str, base_shift: float = 0.5, max_shift: float = 1.15):
        self._base_shift = base_shift
        self._max_shift = max_shift
        self.cfg: float = 2.0,
        self.first_n_steps_without_cfg: int = 4,

    def _time_shift(self, x, t):
        x1, x2 = 256, 4096
        t1, t2 = self._base_shift, self._max_shift
        exp_mu = mx.exp((x - x1) * (t2 - t1) / (x2 - x1) + t1)
        t = exp_mu / (exp_mu + (1 / t - 1))
        return t
    
    def get_lin_function(self,
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b
    
    def time_shift(self, mu: float, sigma: float, t: mx.array):
        return mx.exp(mu) / (mx.exp(mu) + (1 / t - 1) ** sigma)


    @lru_cache
    def timesteps(
        self, num_steps, image_sequence_length, start: float = 1, stop: float = 0
    ):
        t = mx.linspace(start, stop, num_steps + 1)
        # t = self._time_shift(image_sequence_length, t)

        # # chroma
        mu = self.get_lin_function(y1=self._base_shift, y2=self._max_shift)(image_sequence_length)
        t = self.time_shift(mu, 1.0, t)
        return t.tolist()

    def random_timesteps(self, B, L, dtype=mx.float32, key=None):
        # if self._schnell:
            # TODO: Should we upweigh 1 and 0.75?
        t = mx.random.randint(1, 5, shape=(B,), key=key)
        t = t.astype(dtype) / 4

        return t

    def sample_prior(self, shape, dtype=mx.float32, key=None):
        return mx.random.normal(shape, dtype=dtype, key=key)

    def add_noise(self, x, t, noise=None, key=None):
        noise = (
            noise
            if noise is not None
            else mx.random.normal(x.shape, dtype=x.dtype, key=key)
        )
        return x * (1 - t) + t * noise

    def step(self, pred, x_t, t, t_prev):

        return x_t + (t_prev - t) * pred
