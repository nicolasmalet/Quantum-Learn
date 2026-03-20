from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=False)
class QuantumParameters:
    g_conv: float
    g_sq: float

    def as_array(self):
        return np.array([getattr(self, f.name) for f in fields(self)])

    def from_array(self, array: np.ndarray):
        for field, value in zip(fields(self), array):
            setattr(self, field.name, value)

    def __isub__(self, array: np.ndarray):
        for field, value in zip(fields(self), array):
            setattr(self, field.name, getattr(self, field.name) - value)
