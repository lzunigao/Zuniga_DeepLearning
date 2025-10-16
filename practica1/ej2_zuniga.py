#%%
import sklearn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
# import importlib
# import torch.nn as nn
from torch import Tensor
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# %% Nearest Neighbor Classifier

class NearestNeighbor_2:
    def __init__(self):
        self.X = None
        self.Y = None        

    def train(self, X_train, Y_train):
        self.X = X_train.flatten(start_dim=1)
        self.Y = Y_train

    def predict(self, X_test, k = 7):
        
        assert self.X is not None, 'Train the model first'

        Yp = np.zeros(X_test.shape[0])

        for i, x in enumerate(X_test):
            
            norm = np.linalg.norm(np.abs(x.flatten() - self.X), axis=-1, ord=2)
            
            idx_sorted = np.argsort(norm)
            k_nearest = idx_sorted[:k]
            breakpoint()
            Y_preds = self.Y[k_nearest]
            # return the most common class label among the k nearest neighbors
            Yp[i] = np.bincount(Y_preds).argmax()
        Yp = np.array(Yp).astype(int)
        return Yp




#%%    
print(svm.SVC(kernel='linear').get_params())
    
    

# %%
if __name__ == "__main__":        
    # Download and load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    cifar_train = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    cifar_test = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    # %%
    print('Shapes: \n')
    print(f'MNIST train: {mnist_train.data.shape}, test: {mnist_test.data.shape}')
    print(f'MNIST train: {mnist_train.targets.shape}, test: {mnist_test.targets.shape}')
    # %%
    # print(f'CIFAR-10 train: {cifar_train.data.shape}, test: {cifar_test.data.shape}')
    # print(f'CIFAR-10 train: {len(cifar_train.targets)}, test: {len(cifar_test.targets)}')
    
    #%%
    random_range = np.random.randint(low=0, high=mnist_train.data.shape[0], size=2000).astype(int)
    x_train1 = Tensor(mnist_train.data)
    y_train1 = np.array(mnist_train.targets)

    x_test1 = Tensor(mnist_test.data)
    y_test1 = np.array(mnist_test.targets)

    plt.figure(figsize=(12,3))
    for i, num in enumerate(x_test1[:5]):
        plt.subplot(1, 5, i+1)
        plt.imshow(num.squeeze(), cmap='gray')
        plt.title(f'Label: {y_test1[i]}')
        plt.axis('off')

    #%%
    
    
    model = NearestNeighbor_2()
    model.train(x_train1, y_train1)

    k_values = np.array(np.arange(start=1, stop=14, step=2))
    acc1 = []    

    for k_ in k_values:
        y_pred1 = model.predict(x_test1[:20], k=k_)
        
        print(f'Predictions for k={k_}:', y_pred1)        
        print(f'Targets for k={k_}:', y_test1[:20], '\n')        
        acc1.append(100*np.sum((y_pred1 == y_test1[:20]))/20)
      
    
    plt.plot(k_values, acc1, marker='o')
    plt.title('MNIST Accuracy vs k-neighbors')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.grid()

#%%
    names = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
    random_range = np.random.randint(low=0, high=cifar_train.data.shape[0], size=2000).astype(int)
    x_train2 = Tensor(cifar_train.data)
    y_train2 = np.array(cifar_train.targets)

    x_test2 = Tensor(cifar_test.data)
    y_test2 = np.array(cifar_test.targets)

    plt.figure(figsize=(12,3))
    for i, num in enumerate(x_test2[:5]):
        plt.subplot(1, 5, i+1)
        plt.imshow(num.numpy().astype(np.uint8))
        plt.title(f'Label: {names[y_test2[i]]}')
        plt.axis('off')

    plt.show()
    #%%
    
    model = NearestNeighbor_2()
    model.train(x_train2, y_train2)

    k_values = np.array(np.arange(start=1, stop=14, step=2))
    acc2 = []

    for k_ in k_values:
        y_pred2 = model.predict(x_test2[:20], k=k_)

        # print(f'Predictions for k={k_}:', y_pred2)
        # print(f'Targets for k={k_}:', y_test2[:20], '\n')
        acc2.append(100*np.sum((y_pred2 == y_test2[:20]))/20)

    plt.plot(k_values, acc2, marker='o')
    plt.title('CIFAR-10 Accuracy vs k-neighbors')
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    # %%

    # %%
    #%% Support vector machine
    