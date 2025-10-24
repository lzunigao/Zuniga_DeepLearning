#%%
import numpy as np
import activations


class BaseLayer(): #Clase genérica de cualquier tipo de capa sin pesos ni bias
    def __init__(self,
                input_shape,
                output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape       

    def __call__(self, x):        
        if x.shape[-1] != self.input_shape[-1]:
            raise ValueError(f"Dimensiones de entrada incorrectas. Se esperaban {self.input_shape[-1]}, pero se recibieron {x.shape[-1]}")
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
                input_shape):
        super().__init__(input_shape, input_shape)

### SON TIPOS DIFERENTES DE CAPAS ###

class Layer(): #Clase genérica de cualquier tipo de capa con pesos y bias
    
    def __init__(self,
                input_shape,
                output_shape,
                activation=None):
        if isinstance(input_shape, int):
            input_shape = (1, input_shape)
            
        if isinstance(output_shape, int):
            output_shape = (output_shape,1)
                
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.randn(self.input_shape[-1], self.output_shape[0]) * 0.01
        self.bias = np.zeros((input_shape[0], self.output_shape[0])) # batch size x output shape

    def __call__(self, x):
        if x.shape[-1] != self.input_shape[-1]:
            raise ValueError(f"Dimensiones de entrada incorrectas. Se esperaban {self.input_shape}, pero se recibieron {x.shape}")
        return np.dot(x, self.weights) + self.bias

class DenseLayer(Layer): # Capa densa con activación
    def __init__(self, input_shape, output_shape, activation):
        super().__init__(input_shape, output_shape)
        self.activation = activation if activation is not None else lambda x: x
        
    def __call__(self, x):
        z = super().__call__(x)
        return self.activation(z)
    
    def backward(self, grad_output, learning_rate):        
        #TODO: preguntar sobre self.last_input
        # Cálculo del gradiente de la capa
        z = np.dot(self.last_input, self.weights) + self.bias
        grad_activation = self.activation(z, return_derivative=True)
        grad = grad_output * grad_activation
        
        # Gradientes para pesos y bias
        grad_weights = np.dot(self.last_input.T, grad)
        grad_bias = np.sum(grad, axis=0, keepdims=True)
        
        # Actualización de pesos y bias
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        # Retornar el gradiente para la capa anterior
        return np.dot(grad, self.weights.T)
    
#%%    
if __name__=="__main__":
    lay = Layer(
        input_shape=2,
        output_shape=3,
        activation=None
    )

    x = np.random.randn(7,4)
    lay2 = InputLayer_like(x)

    print(lay2.input_shape)
    print(lay2.output_shape)

# %%
