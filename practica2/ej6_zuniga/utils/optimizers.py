
# Optimizador: interfaz para los optimizadores
#SGD: clase que implementa el optimizador stochastic gradient descent

class Optimizador:
    def __init__(self, params, learning_rate):
        self.params = params
        self.learning_rate = learning_rate
        self.grads = [0 for _ in params]

    def step(self, *args, **kwargs):
        for i, param in enumerate(self.params):
            param -= self.learning_rate * self.grads[i]
        self.zero_grad()

    def zero_grad(self):
        for param in self.params:
            param.grad = 0


class SGD(Optimizador): # recorre aleatoriamente los datos y actualiza los pesos
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params, learning_rate)
    