
# Optimizador: interfaz para los optimizadores
#SGD: clase que implementa el optimizador stochastic gradient descent

class Optimizador:
    def __init__(self, learning_rate, params, loss_fct):
        self.params = params
        self.learning_rate = learning_rate
        self.loss_fct = loss_fct

    def step(self):
        for i, param in enumerate(self.params):
            param -= self.learning_rate * self.grads[i]
        self.zero_grad()

    def zero_grad(self):
        for param in self.params:
            param.grad = 0


class SGD(Optimizador): # recorre aleatoriamente los datos y actualiza los pesos
    pass
    