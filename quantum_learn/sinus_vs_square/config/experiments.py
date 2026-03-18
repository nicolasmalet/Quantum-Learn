from dataclasses import dataclass

from zeroth.experiment import ExperimentConfig, VariationConfig

from quantum_learn.sinus_vs_square.data import create_data_sinus_vs_square
from . import paths
from .data_config import NB_POINTS_PER_PERIOD
from .jpc_config import NB_QUADRATURES
from .models import quantum_model_config


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
        values=[[1, NB_QUADRATURES * NB_POINTS_PER_PERIOD * 1, 100],
                [10, NB_QUADRATURES * NB_POINTS_PER_PERIOD * 10, 100],
                [100, NB_QUADRATURES * NB_POINTS_PER_PERIOD * 100, 100]]
    )

    quantum_lr = VariationConfig(
        name="quantum_lr",
        param=[paths.QUANTUM_LR],
        values=[[0.001], [0.01], [0.1], [1]]
    )


VARIATIONS = VariationCatalog()


#  WARNING: some VariationConfig might overwrite others depending on the order of variations
@dataclass(frozen=True)
class ExperimentCatalog:
    measure_resolution: ExperimentConfig = ExperimentConfig(name="measure_resolution",
                                                            title="Measure Resolution",
                                                            base_model=quantum_model_config,
                                                            variations=[VARIATIONS.measure_resolution],
                                                            create_data=create_data_sinus_vs_square,
                                                            plot_dimension=0)

    quantum_lr: ExperimentConfig = ExperimentConfig(name="quantum_lr",
                                                    title="Training Loss vs Chip Parameters LR",
                                                    base_model=quantum_model_config,
                                                    variations=[VARIATIONS.quantum_lr],
                                                    create_data=create_data_sinus_vs_square,
                                                    plot_dimension=1)


EXPERIMENTS = ExperimentCatalog()
