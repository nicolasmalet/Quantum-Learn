from zeroth import first_order
from zeroth.abstract import NeuralNetworkConfig, NetworkArchitecture
from zeroth.utils.activation_functions import Softmax, ReLU


linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    architecture=NetworkArchitecture(hidden_dims=[],
                                     activations=[Softmax()])
)

XS: NeuralNetworkConfig = NeuralNetworkConfig(
    name="xs",
    architecture=NetworkArchitecture(hidden_dims=[10],
                                     activations=[ReLU(), Softmax()])
)

first_order_adam = first_order.FirstOrderAdamConfig(learning_rate=0.02,
                                                    beta1=0.9,
                                                    beta2=0.99,
                                                    epsilon=1e-8)
