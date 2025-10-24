import layers
import numpy as np
import activations
import optimizers

class Network: # Red neuronal que contiene múltiples capas
    def __init__(self):
        raise NotImplementedError("La clase Network debe especificar su propia arquitectura")

    def add_layer(self, input_shape, output_shape, activation):
        if not hasattr(self, 'layers'):
            self.layers = []
        layer = layers.DenseLayer(input_shape, output_shape, activation)
        self.layers.append(layer)

    def compile(self, learning_rate, optimizer): # ... parámetros como learning rate, qué más?
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def forward(self, x):
        raise NotImplementedError("El método forward debe ser implementado en la subclase")

    def fit(self, x, y, epochs, ):
        for epoch in range(epochs):            
            output = self.forward(x)
            # Compute loss (mean squared error)
            loss = np.mean((output - y) ** 2)
            # Backward pass
            grad = 2 * (output - y) / y.shape[0]
            for layer in reversed(self.layers):
                
                grad = layer.backward(grad, learning_rate=0.01)
                
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
class XOR1(Network):
    def __init__(self):        
        self.add_layer(2, 2, activations.tanh)
        self.add_layer(2, 1, activations.tanh)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
  

    
class XOR2(Network):
    def __init__(self):
        self.add_layer(2, 1, activations.tanh)
        self.add_layer(3, 1, activations.tanh)
    def forward(self, x): # me interesa cambiar la arquitectura
        x_hid = self.layers[0](x)
        x_concat = np.concatenate([x, x_hid], axis=1)
        x_out = self.layers[1](x_concat)
        return x_out
    
#%%
