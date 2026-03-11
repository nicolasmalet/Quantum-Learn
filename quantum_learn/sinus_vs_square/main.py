from .data import create_data_sinus_vs_square
from .config import quantum_model_config


def main():
    M = quantum_model_config.instantiate()
    data = create_data_sinus_vs_square(1000, 100)
    M.train(data, 100)
    M.plot_loss()
    M.test(data)
