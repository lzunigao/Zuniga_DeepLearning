from utils import models, layers, activations, optimizers
import numpy as np

class XOR1(models.Network):
    def __init__(self):
        super().__init__()
        self.add_layer(layers.InputLayer(2))
        self.add_layer(layers.DenseLayer(2, 2, activation=activations.tanh))
        self.add_layer(layers.DenseLayer(2, 1, activation=activations.tanh))
    def forward(self, x):
        return super().__call__(x)
    
class XOR2(models.Network):
    def __init__(self):
        super().__init__()
        self.add_layer(layers.InputLayer(2))
        self.add_layer(layers.DenseLayer(2, 1, activation=activations.tanh))
        self.add_layer(layers.DenseLayer(3, 1, activation=activations.tanh))
    def forward(self, x):
        out1 = super().__call__(x)
        out2 = np.hstack((x, out1))
        return self.layers[2](out2)
        
xor = XOR1()
inputs = [[0,0],[0,1],[1,0],[1,1]]
outputs = xor.forward(np.array(inputs))

print("Entradas:")
print(np.array(inputs))
print("Salidas:")
print(outputs)