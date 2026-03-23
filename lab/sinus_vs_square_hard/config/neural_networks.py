from zeroth import first_order
from zeroth.abstract import NeuralNetworkConfig, LayerConfig
from zeroth.utils.activation_functions import softmax, relu

from .data_config import *
from ..task_constants import NB_CLASS
from quantum_simulation.jpc_config import quantum_constants

nb_quadratures = 4
input_dim_quadratures = 4 * measure_resolution
input_dim_photons = quantum_constants.DIM_A * quantum_constants.DIM_B * measure_resolution
output_dim = NB_CLASS

linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    layers_config=[LayerConfig(input_dim=input_dim_quadratures, output_dim=output_dim, f=softmax)]
)

XS: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    layers_config=[LayerConfig(input_dim=input_dim_quadratures, output_dim=input_dim_quadratures, f=relu),
                   LayerConfig(input_dim=input_dim_quadratures, output_dim=output_dim, f=softmax)]
)

linear_photons: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    layers_config=[LayerConfig(input_dim=input_dim_photons, output_dim=output_dim, f=softmax)]
)


first_order_adam = first_order.FirstOrderAdamConfig(learning_rate=0.02,
                                                    beta1=0.9,
                                                    beta2=0.99,
                                                    epsilon=1e-8)
