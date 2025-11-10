#%%%

from utils.models import XOR1, XOR2
from utils import optimizers, losses
import numpy as np
np.set_printoptions(precision=2, suppress=False)

#%%

#%%

if __name__ == "__main__":

    net = XOR1()
    loss_fct = losses.MeanSquaredError()

    net.compile(learning_rate=0.01, optimizer=optimizers.SGD(
        learning_rate=0.01,
        params=net.layers,
        loss_fct=loss_fct
    ))

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    net.fit(X, y, epochs=1)