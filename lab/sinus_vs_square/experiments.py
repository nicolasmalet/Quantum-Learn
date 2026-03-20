from dataclasses import dataclass

from zeroth.experiment import ExperimentConfig, VariationConfig

from lab.sinus_vs_square.create_data import create_data
from quantum_learn import paths
from quantum_simulation.jpc_config import nb_quadratures
from lab.sinus_vs_square.models import quantum_model_config, no_quantum_learning_model
from lab.sinus_vs_square.config.quantum_network_config import null_gradient_estimator
from lab.sinus_vs_square.task_constants import NB_POINTS_PER_PERIOD


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
        values=[[0.001], [0.01], [0.1], [1]]
    )

    null_gradient = VariationConfig(
        name="No Quantum Learning",
        param=[paths.GRADIENT_ESTIMATOR],
        values=[[null_gradient_estimator]]
    )


VARIATIONS = VariationCatalog()


#  WARNING: some VariationConfig might overwrite others depending on the order of variations
@dataclass(frozen=True)
class ExperimentCatalog:
    test_experiment: ExperimentConfig = ExperimentConfig(name="test_experiment",
                                                         title="Measure Resolution",
                                                         base_model=quantum_model_config,
                                                         variations=[],
                                                         create_data=create_data,
                                                         plot_dimension=0,
                                                         smooth_fraction=0)

    measure_resolution: ExperimentConfig = ExperimentConfig(name="measure_resolution",
                                                            title="Measure Resolution",
                                                            base_model=quantum_model_config,
                                                            variations=[VARIATIONS.measure_resolution],
                                                            create_data=create_data,
                                                            plot_dimension=0,
                                                            smooth_fraction=0)

    quantum_lr: ExperimentConfig = ExperimentConfig(name="quantum_lr",
                                                    title="Training Loss vs Chip Parameters LR",
                                                    base_model=quantum_model_config,
                                                    variations=[VARIATIONS.quantum_lr],
                                                    create_data=create_data,
                                                    plot_dimension=1,
                                                    smooth_fraction=0)

    no_quantum_learning: ExperimentConfig = ExperimentConfig(name="no_quantum_learning",
                                                             title="Training loss vs no quantum learning",
                                                             base_model=no_quantum_learning_model,
                                                             variations=[],
                                                             create_data=create_data,
                                                             plot_dimension=0,
                                                             smooth_fraction=0)


EXPERIMENTS = ExperimentCatalog()
