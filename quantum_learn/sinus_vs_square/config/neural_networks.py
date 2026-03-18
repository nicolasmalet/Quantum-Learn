from zeroth import first_order
from zeroth.abstract import NeuralNetworkConfig, LayerConfig
from zeroth.utils.activation_functions import softmax

from .data_config import *
from .jpc_config import NB_QUADRATURES

INPUT_DIM = NB_QUADRATURES * NB_POINTS_PER_PERIOD * MEASURE_RESOLUTION
OUTPUT_DIM = NB_CLASS

linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    layers_config=[LayerConfig(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, f=softmax)]
)

first_order_adam = first_order.FirstOrderAdamConfig(learning_rate=0.02,
                                                    beta1=0.9,
                                                    beta2=0.99,
                                                    epsilon=1e-8)
