#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%%

train_data = pd.read_csv('./logs_ej2.txt', delimiter=',')
# %%
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_data['Epoch'], train_data['Loss'])
plt.savefig('ej2_loss_plot.png', dpi=300)
plt.show()


plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_data['Epoch'], train_data['Acc'])
plt.savefig('ej2_accuracy_plot.png', dpi=300)
plt.show()
# %%
