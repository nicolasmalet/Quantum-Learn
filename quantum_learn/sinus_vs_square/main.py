from .data import create_data_sinus_vs_square
from .config import quantum_model_config
from .config.data_config import *


def main():
    M = quantum_model_config.instantiate()
    data = create_data_sinus_vs_square(NB_PERIOD_TRAIN, NB_PERIOD_TEST, NB_POINTS_PER_PERIOD)
    M.train(data, NB_PERIOD_TRAIN // BATCH_SIZE)
    M.plot_loss()
    M.test(data)
