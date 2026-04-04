from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from zeroth.abstract import Summary


@dataclass(frozen=False)
class QuantumParameters(Summary):
    g_conv_real: float
    g_sq_real: float
    g_conv_imag: float
    g_sq_imag: float
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


@dataclass(frozen=True, kw_only=True)
class QuantumParametersConfig(Summary):
    g_conv_real: float
    g_sq_real: float
    g_conv_imag: float = 0
    g_sq_imag: float = 0
    delta_a: float = 0
    delta_b: float = 0

    @classmethod
    def get_indices(cls, *names: str) -> list[int]:
        all_fields = {f.name: i for i, f in enumerate(fields(cls))}
        return [all_fields[name] for name in names]

    def instantiate(self) -> QuantumParameters:
        return QuantumParameters(g_conv_real=self.g_conv_real,
                                 g_sq_real=self.g_sq_real,
                                 g_conv_imag=self.g_conv_imag,
                                 g_sq_imag=self.g_sq_imag,
                                 delta_a=self.delta_a,
                                 delta_b=self.delta_b)
