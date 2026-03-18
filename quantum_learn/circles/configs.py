import numpy as np
from zeroth import first_order
from zeroth.abstract import NeuralNetworkConfig, LayerConfig
from zeroth.utils.activation_functions import softmax
from zeroth.zeroth_order import ZerothOrderAdamConfig
from zeroth.zeroth_order.gradient_estimators import FiniteDifferenceConfig

from quantum_learn.quantum_black_box import QuantumBlackBoxConfig

INPUT_DIM = 8
OUTPUT_DIM = 2
DEFAULT_PERTURBATION_SCALE = 0.01

quantum_network_config = QuantumBlackBoxConfig(
    name="Q_Network",
    quantum_params=np.array([50.0, 50.0]),
)

first_order_adam = first_order.FirstOrderAdamConfig(learning_rate=0.02,
                                                    beta1=0.9,
                                                    beta2=0.99,
                                                    epsilon=1e-8)

zeroth_order_adam = ZerothOrderAdamConfig(learning_rate=0.02,
                                          beta1=0.9,
                                          beta2=0.99,
                                          epsilon=1e-8)

finite_difference: FiniteDifferenceConfig = FiniteDifferenceConfig(dA=DEFAULT_PERTURBATION_SCALE)

linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    layers_config=[LayerConfig(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, f=softmax)]
)
