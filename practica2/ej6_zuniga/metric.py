import numpy as np

def accuracy(y_true, y_pred): #accuracy del XOR. Las salidas tienen un solo bit
    return np.mean(y_true == y_pred, axis=-1)

def mse(y_true, y_pred): #mean squared error
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must be the same."
    return np.mean((y_true - y_pred) ** 2, axis=-1)