import numpy as np

def accuracy(y_true, y_pred): #accuracy del XOR. Las salidas tienen un solo bit
    return np.round( np.mean(y_true == y_pred, axis=-1) , 3)

def mse(y_true, y_pred): #mean squared error
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must be the same."
    return np.round(np.mean((y_true - y_pred) ** 2, axis=-1), 3)

if __name__ == "__main__":
    # Ejemplo de uso
    y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    y_pred = np.array([[0.9, 0.1, 0.8, 0.2], [0.2, 0.8, 0.1, 0.9]])
    
    acc = accuracy(y_true, np.round(y_pred))
    mse_value = mse(y_true, y_pred)
    
    print("Accuracy:", acc)
    print("Mean Squared Error:", mse_value)