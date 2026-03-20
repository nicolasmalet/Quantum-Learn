from zeroth import first_order
from zeroth.abstract import NeuralNetworkConfig, LayerConfig
from zeroth.utils.activation_functions import softmax

from .data_config import *
from .jpc_config import nb_quadratures
from ..task_constants import NB_POINTS_PER_PERIOD, NB_CLASS

input_dim = nb_quadratures * NB_POINTS_PER_PERIOD * measure_resolution
output_dim = NB_CLASS

linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    layers_config=[LayerConfig(input_dim=input_dim, output_dim=output_dim, f=softmax)]
)

first_order_adam = first_order.FirstOrderAdamConfig(learning_rate=0.02,
                                                    beta1=0.9,
                                                    beta2=0.99,
                                                    epsilon=1e-8)
