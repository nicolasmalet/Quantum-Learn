from dataclasses import dataclass

import jax.numpy as jnp
from zeroth.abstract.summary import Summary

from .quantum_parameters import QuantumParameters
from ..encoding_functions import EncodingFunction


@dataclass(frozen=True)
class Encoding(Summary):
    encoding_parameters: tuple
    encoding_function: EncodingFunction

    def get_value(self, quantum_parameters: QuantumParameters, name: str, data: jnp.ndarray) -> jnp.ndarray:
        base_value: float = getattr(quantum_parameters, name)

        if name in self.encoding_parameters:
            return self.encoding_function(base_value, data)

        return base_value * jnp.ones_like(data)
