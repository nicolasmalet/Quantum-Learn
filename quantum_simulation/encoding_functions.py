from abc import ABC, abstractmethod
from typing import override

import jax.numpy as jnp


class EncodingFunction(ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, quantum_parameter: float, X: jnp.ndarray) -> jnp.ndarray:
        ...


class Linear(EncodingFunction):
    @override
    def __call__(self, quantum_parameter: float, X: jnp.ndarray) -> jnp.ndarray:
        return quantum_parameter * X


class Affine(EncodingFunction):
    def __init__(self, bias: float = 0):
        self.bias: float = bias

    @override
    def __call__(self, quantum_parameter: float, X: jnp.ndarray) -> jnp.ndarray:
        return (quantum_parameter + self.bias) * X
