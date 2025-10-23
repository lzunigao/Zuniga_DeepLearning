#%%
import numpy as np
class BaseLayer: # CLase genérica de cualquier tipo de capa
        
    def __init__(self, input_dims, output_dims): 
        self.input_dims = input_dims
        self.output_dims = output_dims
    def __call__(self, x):
        raise NotImplementedError("Cada capa debe implementar su propio método __call__")
        
class InputLayer(BaseLayer): # Capa de entrada    
    def __init__(self, input_dims):
        super().__init__(input_dims, input_dims)        
    def __call__(self, x):
        if x.shape[1] != self.input_dims:
            raise ValueError(f"Dimensiones de entrada incorrectas. Se esperaban {self.input_dims}, pero se recibieron {x.shape[1]}")
        return x

class Layer(BaseLayer): #Capa genérica con pesos y bias
    def __init__(self, input_dims, output_dims):
        super().__init__(input_dims, output_dims)
        self.weights = np.random.randn(input_dims, output_dims) * 0.01
        self.bias = np.zeros((1, output_dims))
        
    def __call__(self, x):
        if x.shape[1] != self.input_dims:
            raise ValueError(f"Dimensiones de entrada incorrectas. Se esperaban {self.input_dims}, pero se recibieron {x.shape[1]}")
        return np.dot(x, self.weights) + self.bias

class DenseLayer(Layer): # Capa densa con activación
    def __init__(self, input_dims, output_dims, activation):
        super().__init__(input_dims, output_dims)
        self.activation = activation
        
    def __call__(self, x):
        z = super().__call__(x)
        return self.activation(z)
#%%    


    
