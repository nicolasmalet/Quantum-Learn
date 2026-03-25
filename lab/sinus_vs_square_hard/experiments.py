from dataclasses import dataclass

from zeroth.experiment import ExperimentConfig, VariationConfig

from quantum_learn import paths
from . import models
from .config import neural_networks as nn
from .config.quantum_network_config import null_gradient_estimator
from .data import create_data_default
from .task_constants import NB_POINTS_PER_PERIOD

nb_quadratures = 4


class VariationCatalog:
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


VARIATIONS = VariationCatalog()

SMOOTH_FRACTION = 0.05


#  WARNING: some VariationConfig might overwrite others depending on the order of variations
@dataclass(frozen=True)
class ExperimentCatalog:
    test_experiment: ExperimentConfig = ExperimentConfig(name="test_experiment",
                                                         title="Measure Resolution",
                                                         base_model=models.quantum_model_config,
                                                         variations=[],
                                                         create_data=create_data_default,
                                                         plot_dimension=0,
                                                         smooth_fraction=SMOOTH_FRACTION)

    measure_resolution: ExperimentConfig = ExperimentConfig(name="measure_resolution",
                                                            title="Measure Resolution",
                                                            base_model=models.quantum_model_config,
                                                            variations=[VARIATIONS.measure_resolution],
                                                            create_data=create_data_default,
                                                            plot_dimension=0,
                                                            smooth_fraction=SMOOTH_FRACTION)

    quantum_lr: ExperimentConfig = ExperimentConfig(name="quantum_lr",
                                                    title="Training Loss vs Chip Parameters LR",
                                                    base_model=models.quantum_model_config,
                                                    variations=[VARIATIONS.quantum_lr],
                                                    create_data=create_data_default,
                                                    plot_dimension=1,
                                                    smooth_fraction=SMOOTH_FRACTION)

    no_quantum_learning: ExperimentConfig = ExperimentConfig(name="no_quantum_learning",
                                                             title="Training loss vs no quantum learning",
                                                             base_model=models.no_quantum_learning_model_xs,
                                                             variations=[],
                                                             create_data=create_data_default,
                                                             plot_dimension=0,
                                                             smooth_fraction=SMOOTH_FRACTION)

    no_quantum_learning_vs_nn_sizes = ExperimentConfig(name="no_quantum_learning_vs_nn_sizes",
                                                       title="Training loss, no quantum learning",
                                                       base_model=models.no_quantum_learning_model_xs,
                                                       variations=[VARIATIONS.classical_network_size],
                                                       create_data=create_data_default,
                                                       plot_dimension=0,
                                                       smooth_fraction=SMOOTH_FRACTION)

    photon: ExperimentConfig = ExperimentConfig(name="photon",
                                                title="photon model",
                                                base_model=models.photon_model_config,
                                                variations=[],
                                                create_data=create_data_default,
                                                plot_dimension=0,
                                                smooth_fraction=SMOOTH_FRACTION)

    dudas: ExperimentConfig = ExperimentConfig(name="dudas",
                                                title="Training Loss Dudas",
                                                base_model=models.quantum_model_config_dudas,
                                                variations=[],
                                                create_data=create_data_default,
                                                plot_dimension=0,
                                                smooth_fraction=SMOOTH_FRACTION)


EXPERIMENTS = ExperimentCatalog()
