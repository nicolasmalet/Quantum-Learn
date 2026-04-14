from zeroth.experiment import ExperimentConfig

from ..config import models, variations
from .data import data_creator

SMOOTH_FRACTION = 0.05

test_experiment: ExperimentConfig = ExperimentConfig(name="test_experiment",
                                                     base_model=models.quantum_model_config,
                                                     variations=[],
                                                     data_creator=data_creator)

quantum_lr: ExperimentConfig = ExperimentConfig(name="quantum_lr",
                                                base_model=models.quantum_model_config,
                                                variations=[variations.quantum_lr],
                                                data_creator=data_creator)

dudas: ExperimentConfig = ExperimentConfig(name="dudas",
                                           base_model=models.quantum_model_config_dudas,
                                           variations=[variations.null_gradient],
                                           data_creator=data_creator)
