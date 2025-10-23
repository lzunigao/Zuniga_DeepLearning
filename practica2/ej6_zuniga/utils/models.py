import layers
import activations
import numpy as np

class Network: # Red neuronal que contiene múltiples capas
    def __init__(self):
        raise NotImplementedError("La clase Network debe especificar su propia arquitectura")

    def forward(self, x):
        raise NotImplementedError("La clase Network debe implementar su propio método forward")
    
    def train(self):
        # permite hacer backpropagation
        pass
    def test(self):
        # permite hacer inferencia
        pass

    def backpropagation(self):
        # calcular gradientes y actualizar pesos
        pass
    
class XOR1(Network):
    def __init__(self):        
        self.input = layers.InputLayer(2)
        self.hidden = layers.DenseLayer(2, 2, activation=activations.tanh)
        self.output = layers.DenseLayer(2, 1, activation=activations.tanh)
    
    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        return x

    
class XOR2(Network):
    def __init__(self):
        super().__init__()
        self.input = layers.InputLayer(2)
        self.hidden = layers.DenseLayer(2, 1, activation=activations.tanh)
        self.output = layers.DenseLayer(3, 1, activation=activations.tanh)
    def forward(self, x):
        x_in = self.input(x)
        x_hidden = self.hidden(x_in)
        x_concat = np.concatenate([x_in, x_hidden], axis=1)
        x_out = self.output(x_concat)
        return x_out