from zeroth import first_order
from zeroth.abstract import NeuralNetworkConfig
from zeroth.utils.activation_functions import Softmax, ReLU, Sigmoid

linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    hidden_dims=[],
    activations=[Sigmoid()]
)

XS: NeuralNetworkConfig = NeuralNetworkConfig(
    name="xs",
    hidden_dims=[10],
    activations=[ReLU(), Sigmoid()]
)

first_order_adam = first_order.FirstOrderAdamConfig(learning_rate=0.02,
                                                    beta1=0.9,
                                                    beta2=0.99,
                                                    epsilon=1e-8)
