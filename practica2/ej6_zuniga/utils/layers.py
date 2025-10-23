#%%
from typing import Tuple
import numpy as np


class BaseLayer(): #Clase genérica de cualquier tipo de capa sin pesos ni bias
    def __init__(self,
                input_shape: Tuple[int, ...],
                output_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        self.output_shape = output_shape       

    def __call__(self, x):        
        if x.shape != self.input_shape:
            raise ValueError(f"Dimensiones de entrada incorrectas. Se esperaban {self.input_shape}, pero se recibieron {x.shape}")
        return x

class InputLayer_like(BaseLayer): # Capa de entrada
    def __init__(self, example):
        if isinstance(example, np.ndarray):
            example_shape = example.shape
        elif isinstance(example, list):
            example_shape = (len(example),)
        else:
            raise ValueError("El ejemplo debe ser un ndarray de numpy o una lista")
        super().__init__(example_shape, example_shape)

class InputLayer(BaseLayer): # Capa de entrada
    def __init__(self,                  
                input_shape: Tuple[int, ...]):
        super().__init__(input_shape, input_shape)

### SON TIPOS DIFERENTES DE CAPAS ###

class Layer(): #Clase genérica de cualquier tipo de capa con pesos y bias
    def __init__(self,
                input_shape: Tuple[int, ...],
                output_shape: Tuple[int, ...],
                activation=None):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(self.input_shape[-1], self.output_shape[0]) * 0.01
        self.bias = np.zeros(self.output_shape) 

    def __call__(self, x):
        if x.shape != self.input_shape:
            raise ValueError(f"Dimensiones de entrada incorrectas. Se esperaban {self.input_shape}, pero se recibieron {x.shape}")
        return np.dot(x, self.weights) + self.bias

class DenseLayer(Layer): # Capa densa con activación
    def __init__(self, input_shape, output_shape, activation):
        super().__init__(input_shape, output_shape)
        self.activation = activation if activation is not None else lambda x: x
        
    def __call__(self, x):
        z = super().__call__(x)
        return self.activation(z)
#%%    
if __name__=="__main__":
    lay = Layer(
        input_shape=(1,2),
        output_shape=(3,1),
        activation=None
    )
    print(lay.weights)

# %%
