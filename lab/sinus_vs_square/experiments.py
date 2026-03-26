from zeroth.experiment import ExperimentConfig

from . import models, variations
from .data import create_data_default

SMOOTH_FRACTION = 0.05

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
                                                        variations=[variations.measure_resolution],
                                                        create_data=create_data_default,
                                                        plot_dimension=0,
                                                        smooth_fraction=SMOOTH_FRACTION)

quantum_lr: ExperimentConfig = ExperimentConfig(name="quantum_lr",
                                                title="Training Loss vs Chip Parameters LR",
                                                base_model=models.quantum_model_config,
                                                variations=[variations.quantum_lr],
                                                create_data=create_data_default,
                                                plot_dimension=1,
                                                smooth_fraction=SMOOTH_FRACTION)

no_quantum_learning: ExperimentConfig = ExperimentConfig(name="no_quantum_learning",
                                                         title="Training loss vs no quantum learning",
                                                         base_model=models.no_quantum_learning_model,
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
