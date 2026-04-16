from __future__ import annotations

from dataclasses import dataclass, fields

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
