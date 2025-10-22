import numpy as np

class Loss:
    def __init__(self):
        pass
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Subclasses should implement this method.")
    def gradient(self, y_true, y_pred):
        raise NotImplementedError("Subclasses should implement this method.")
    
class MeanSquaredError(Loss):
    def __call__(self, y_true, y_pred):
        return np.sqrt(np.sum((y_true - y_pred)**2))
    
    def gradient(self, y_true, y_pred):

        for yt, yp in zip(y_true, y_pred):
            yield -2 * (yt - yp) / np.sqrt(np.sum((y_true - y_pred)**2))

        