#%% 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#%%

train_data = {'class': [], 'vx': [], 'vy': []}
test_data = {'class': [], 'vx': [], 'vy': []}

for i in range(100): 
    k = np.random.choice(np.arange(1,6))
    vx = k + np.round(np.random.randn()/3, 2) + np.random.choice([0, 1])
    vy = k +  np.round(np.random.randn()/3, 2) + np.random.choice([0, 1])




    train_data['class'].append(k)
    train_data['vx'].append(vx)
    train_data['vy'].append(vy)

    test_data['class'].append(k)
    test_data['vx'].append(vx - np.random.randn()/2)
    test_data['vy'].append(vy + np.random.randn()/2)

for key in train_data.keys():
    train_data[key] = np.array(train_data[key])
    test_data[key] = np.array(test_data[key])

train_ds = pd.DataFrame.from_dict(train_data)
test_ds = pd.DataFrame.from_dict(test_data)

print(test_ds)
# %%

class NearestNeighbor:

    def __init__(self, k=7):
        self.X = None
        self.Y = None
        self.k = k # k vecinos

    def train(self, X, Y):
        
        self.X = X
        self.Y = Y

    def predict(self, X):
        assert self.X is not None, 'Train the model first'
        Yp = np.zeros(X.shape[0])
        for idx in range(X.shape[0]):

            norm = np.linalg.norm(self.X - X[idx], axis=-1)
            
            idx_sorted = np.argsort(norm)
            k_nearest = idx_sorted[:self.k]



            Y_preds = self.Y[k_nearest]
            Yp[idx] = np.bincount(Y_preds.astype(int)).argmax()

        return Yp
    
    
# %%

model1 = NearestNeighbor(k=1)
model1.train(train_ds[['vx', 'vy']].values, train_ds['class'].values)
yp1 = model1.predict(test_ds[['vx', 'vy']].values)

model3 = NearestNeighbor(k=3)
model3.train(train_ds[['vx', 'vy']].values, train_ds['class'].values)
yp3 = model3.predict(test_ds[['vx', 'vy']].values)

model7 = NearestNeighbor(k=7)
model7.train(train_ds[['vx', 'vy']].values, train_ds['class'].values)
yp7 = model7.predict(test_ds[['vx', 'vy']].values)





# %%
fig, ax = plt.subplots(2, 2, figsize=(8,8))



scatter0 = ax[0].scatter(train_ds['vx'], train_ds['vy'], c=train_ds['class'], cmap='jet')
legend2 = ax[0].legend(*scatter0.legend_elements(), title="Classes")
ax[0].add_artist(legend2)
ax[0].set_title('Training data classes')

scatter1 = ax[1].scatter(test_ds['vx'], test_ds['vy'], c=yp1, cmap='jet')
real_scatter = ax[1].scatter(test_ds['vx'], test_ds['vy'], c=test_ds['class'], cmap='jet', marker='x')
legend1 = ax[1].legend(*scatter1.legend_elements(), title="Classes")
ax[1].add_artist(legend1)
ax[1].set_title('Predicted classes (k=1)')

scatter3 = ax[2].scatter(test_ds['vx'], test_ds['vy'], c=yp3, cmap='jet')
real_scatter = ax[2].scatter(test_ds['vx'], test_ds['vy'], c=test_ds['class'], cmap='jet', marker='x')
legend1 = ax[2].legend(*scatter3.legend_elements(), title="Classes")
ax[2].add_artist(legend1)
ax[2].set_title('Predicted classes (k=3)')

scatter7 = ax[3].scatter(test_ds['vx'], test_ds['vy'], c=yp7, cmap='jet')
real_scatter = ax[3].scatter(test_ds['vx'], test_ds['vy'], c=test_ds['class'], cmap='jet', marker='x')
legend1 = ax[3].legend(*scatter7.legend_elements(), title="Classes")
ax[3].add_artist(legend1)
ax[3].set_title('Predicted classes (k=7)')

# %%
