from zeroth.experiment import ExperimentConfig

from .data import data_creator
from ..config import models, variations

SMOOTH_FRACTION = 0.05

test_experiment = ExperimentConfig(name="test_experiment",
                                   base_model=models.quantum_model_config,
                                   variations=[],
                                   data_creator=data_creator
                                   )

builf_f_experiment = ExperimentConfig(name="test_experiment",
                                      base_model=models.base_model,
                                      variations=[variations.build_f],
                                      data_creator=data_creator
                                      )
