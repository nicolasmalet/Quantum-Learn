from zeroth.experiment import VariationConfig

from quantum_learn import paths
from .config import neural_networks as nn
from .config.quantum_network_config import null_gradient_estimator, global_finite_difference
from .task_constants import NB_POINTS_PER_PERIOD

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

measure_resolution = VariationConfig(
    name="Measure Resolution",
    param=[paths.MEASURE_RESOLUTION, paths.INPUT_DIM, paths.SIMULATION_RESOLUTION],
    values=[[1, nb_quadratures * NB_POINTS_PER_PERIOD * 1, 100],
            [10, nb_quadratures * NB_POINTS_PER_PERIOD * 10, 100],
            [100, nb_quadratures * NB_POINTS_PER_PERIOD * 100, 100]]
)

quantum_lr = VariationConfig(
    name="quantum_lr",
    param=[paths.QUANTUM_LR],
    values=[[0], [0.01], [0.03], [0.1], [0.3], [1]]
)

null_gradient = VariationConfig(
    name="No Quantum Learning",
    param=[paths.GRADIENT_ESTIMATOR],
    values=[[null_gradient_estimator]]
)

classical_network_size = VariationConfig(
    name="Classical Network Size",
    param=[paths.NN_CONFIG],
    values=[[nn.linear], [nn.XS]]
)

gradient_vs_null_gradient = VariationConfig(
    name="No Quantum Learning",
    param=[paths.GRADIENT_ESTIMATOR],
    values=[[global_finite_difference], [null_gradient_estimator]]
)
