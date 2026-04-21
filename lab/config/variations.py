from zeroth.experiment import VariationConfig

from lab.config import neural_networks as nn
from lab.config.quantum_network_config import null_gradient_estimator, global_finite_difference, \
    partial_gs_finite_difference
from quantum_learn import paths
from lab.config import quantum_network_config as qn
from quantum_learn import build_f
classical_lr = VariationConfig(
    name="Learning Rate",
    param=[paths.CLASSICAL_LR],
    values=[[0.05], [0.1], [0.5], [1]]
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

gradient_estimators = VariationConfig(
    name="No Quantum Learning",
    param=[paths.GRADIENT_ESTIMATOR],
    values=[[null_gradient_estimator], [partial_gs_finite_difference], [global_finite_difference]]
)
build_f = VariationConfig(
    name="F",
    param=[paths.BUILD_F],
    values=[[build_f.BuildFQuadraturesConfig()],
            [build_f.BuildFQuadraturesPolynomialsConfig()],
            [build_f.BuildFProbasConfig(clip_probas=2)]]
)