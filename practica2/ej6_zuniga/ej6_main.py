#%%%
import importlib
import utils
importlib.reload(utils)

from utils.models import XOR1, XOR2
from utils import optimizers, losses
import numpy as np
np.set_printoptions(precision=2, suppress=False)

#%%

#%%
        
xor1 = XOR1()
xor2 = XOR2()

train_X = np.array([[0,0],[0,1],[1,0],[1,1]])
train_Y = np.array([[0],[1],[1],[0]])


# %%
