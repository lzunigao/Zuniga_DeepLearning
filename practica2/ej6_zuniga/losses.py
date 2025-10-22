import numpy as np

class Loss:
    def __init__(self, y_true=None, y_pred=None):
        self.y_true = y_true
        self.y_pred = y_pred

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Definir __call__")
    def gradient(self, *args, **kwargs):
        raise NotImplementedError("Definir gradiente")
    
class MeanSquaredError(Loss):
    def __call__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.round( np.mean((y_true - y_pred) ** 2), 3)
    
    def gradient(self):
        return -2 * (self.y_true - self.y_pred) / self.y_true.size

if __name__ == "__main__":
    # Ejemplo de uso
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    
    mse_loss = MeanSquaredError()
    loss_value = mse_loss(y_true, y_pred)
    loss_gradient = mse_loss.gradient()
    
    print("Mean Squared Error:", loss_value)
    print("Gradient:", loss_gradient)