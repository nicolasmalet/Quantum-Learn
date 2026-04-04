from zeroth.experiment import ExperimentConfig

from . import models, variations
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

no_quantum_learning_xs: ExperimentConfig = ExperimentConfig(name="no_quantum_learning",
                                                         title="Training loss vs no quantum learning",
                                                         base_model=models.no_quantum_learning_model_xs,
                                                         variations=[],
                                                         data_creator=data_creator,
                                                         plot_dimension=0,
                                                         smooth_fraction=SMOOTH_FRACTION)

no_quantum_learning_vs_nn_sizes = ExperimentConfig(name="no_quantum_learning_vs_nn_sizes",
                                                   title="Training loss, no quantum learning",
                                                   base_model=models.no_quantum_learning_model_xs,
                                                   variations=[variations.classical_network_size],
                                                   data_creator=data_creator,
                                                   plot_dimension=0,
                                                   smooth_fraction=SMOOTH_FRACTION)

photon: ExperimentConfig = ExperimentConfig(name="photon",
                                            title="photon model",
                                            base_model=models.photon_model_config,
                                            variations=[],
                                            data_creator=data_creator,
                                            plot_dimension=0,
                                            smooth_fraction=SMOOTH_FRACTION)

dudas: ExperimentConfig = ExperimentConfig(name="dudas",
                                            title="Training Loss Dudas",
                                            base_model=models.quantum_model_config_dudas,
                                            variations=[],
                                            data_creator=data_creator,
                                            plot_dimension=0,
                                            smooth_fraction=SMOOTH_FRACTION)

dudas_network_size: ExperimentConfig = ExperimentConfig(name="dudas_network_size",
                                            title="Training Loss Dudas",
                                            base_model=models.quantum_model_config_dudas,
                                            variations=[variations.classical_network_size],
                                            data_creator=data_creator,
                                            plot_dimension=0,
                                            smooth_fraction=SMOOTH_FRACTION)

dudas_train_all_params: ExperimentConfig = ExperimentConfig(name="dudas_all_params",
                                            title="Training Loss Dudas",
                                            base_model=models.quantum_model_config_dudas_train_all,
                                            variations=[],
                                            data_creator=data_creator,
                                            plot_dimension=0,
                                            smooth_fraction=SMOOTH_FRACTION)

no_quantum_learning_vs_quantum_learning_dudas = ExperimentConfig(name="dudas",
                                                                 title="quantum learning vs not",
                                                                 base_model=models.quantum_model_config_dudas,
                                                                 variations=[variations.gradient_vs_null_gradient],
                                                                 data_creator=data_creator,
                                                                 plot_dimension=0,
                                                                 smooth_fraction=SMOOTH_FRACTION)
