from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import jax.numpy as jnp
import numpy as np
from zeroth.abstract import Summary


@dataclass(frozen=False)
class QuantumParameters(Summary):
    g_conv: float
    g_sq: float
    epsilon_a: float
    epsilon_b: float
    delta_a: float
    delta_b: float

    encoding: tuple[str] = ("g_sq",)

    f_encoding: Callable[[float, jnp.ndarray], jnp.ndarray] = staticmethod(lambda X, data: data * X)

    def get_value(self, name: str, data: jnp.ndarray) -> jnp.ndarray:
        base_value = getattr(self, name)

        if name in self.encoding:
            return self.f_encoding(base_value, data)

        return base_value * jnp.ones_like(data)

    def as_array(self):
        return np.array([getattr(self, f.name) for f in fields(self)])

    def from_array(self, array: np.ndarray):
        for field, value in zip(fields(self), array):
            setattr(self, field.name, value)

    @classmethod
    def get_indices(cls, *names: str) -> list[int]:
        all_fields = {f.name: i for i, f in enumerate(fields(cls))}
        return [all_fields[name] for name in names]

    def __isub__(self, array: np.ndarray) -> QuantumParameters:
        for field, value in zip(fields(self), array):
            setattr(self, field.name, getattr(self, field.name) - value)
        return self
