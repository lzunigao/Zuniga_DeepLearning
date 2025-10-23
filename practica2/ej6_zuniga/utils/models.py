
class Network: # Red neuronal que contiene múltiples capas
    def __init__(self):
        self.layers = []
    def forward(self, x):
        raise NotImplementedError("La clase Network debe implementar su propio método forward")

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x