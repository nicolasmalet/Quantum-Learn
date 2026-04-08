from zeroth.experiment import ExperimentConfig

from ..config import models, variations
from .data import data_creator

SMOOTH_FRACTION = 0.05

test_experiment: ExperimentConfig = ExperimentConfig(name="test_experiment",
                                                     title="Measure Resolution",
                                                     base_model=models.quantum_model_config,
                                                     variations=[],
                                                     data_creator=data_creator,
                                                     plot_dimension=0,
                                                     smooth_fraction=SMOOTH_FRACTION)

quantum_lr: ExperimentConfig = ExperimentConfig(name="quantum_lr",
                                                title="Training Loss vs Chip Parameters LR",
                                                base_model=models.quantum_model_config,
                                                variations=[variations.quantum_lr],
                                                data_creator=data_creator,
                                                plot_dimension=1,
                                                smooth_fraction=SMOOTH_FRACTION)

dudas: ExperimentConfig = ExperimentConfig(name="dudas",
                                           title="Training Loss Dudas",
                                           base_model=models.quantum_model_config_dudas,
                                           variations=[],
                                           data_creator=data_creator,
                                           plot_dimension=0,
                                           smooth_fraction=SMOOTH_FRACTION)
