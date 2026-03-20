from typing import TypeAlias, Callable

import numpy as np

from quantum_simulation.history import StateHistory
from quantum_simulation.parameters_and_constants import SimulationConstants

Array: TypeAlias = np.typing.NDArray[np.float64]
BuildF: TypeAlias = Callable[[StateHistory, SimulationConstants], np.ndarray]