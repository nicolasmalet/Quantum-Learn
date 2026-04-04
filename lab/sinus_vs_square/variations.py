from zeroth.experiment import VariationConfig

from lab.sinus_vs_square.task_constants import NB_POINTS_PER_PERIOD
from quantum_learn import paths
from .config.quantum_network_config import null_gradient_estimator

nb_quadratures = 4

classical_lr = VariationConfig(
    name="Learning Rate",
    param=[paths.CLASSICAL_LR],
    values=[[0.05], [0.1], [0.5], [1]]
)

batch_sizes = VariationConfig(
    name="Batch Size",
    param=[paths.BATCH_SIZE],
    values=[[3], [10], [30]]
)

quantum_lr = VariationConfig(
    name="quantum_lr",
    param=[paths.QUANTUM_LR],
    values=[[0.001], [0.01], [0.1], [1]]
)

null_gradient = VariationConfig(
    name="No Quantum Learning",
    param=[paths.GRADIENT_ESTIMATOR],
    values=[[null_gradient_estimator]]
)
