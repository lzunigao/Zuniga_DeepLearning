#%%
from torchvision import datasets, transforms
import ej1_zuniga
import numpy as np
import matplotlib.pyplot as plt
import importlib
importlib.reload(ej1_zuniga)
from ej1_zuniga import NearestNeighbor
from sklearn import svm
import torch.nn as nn
# %%

class Image_classifier(NearestNeighbor):
    def __init__(self):
        super().__init__()

    def train(self,X, Y_true):
        # Flatten the images
        self.im_shape = X.shape[1:] # (28, 28)
        self.X = np.reshape(X, (X.shape[0], np.prod(self.im_shape))) # N_images x (28*28)
        self.Y = Y_true

class SVM_sklearn:
    def __init__(self):
        self.model = None
        self.im_shape = None

    def train(self, X, Y):
        # Flatten images if needed
        self.im_shape = X.shape[1:]
        X_flat = np.reshape(X, (X.shape[0], np.prod(self.im_shape)))
        self.model = svm.SVC(kernel='linear')
        self.model.fit(X_flat, Y)

    def predict(self, X):
        X_flat = np.reshape(X, (X.shape[0], np.prod(self.im_shape)))
        return self.model.predict(X_flat)
    
class SVM_classifier(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(28*28, 10) # Example for MNIST
        pass
        
    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

    def loss_gradient(self, X, Y):
        pass

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
    print(f'CIFAR-10 train: {cifar_train.data.shape}, test: {cifar_test.data.shape}')
    # %%

    random_range = np.random.choice(mnist_test.data.shape[0], size=20, replace=False)

    x_train1 = mnist_train.data.numpy()
    y_train1 = mnist_train.targets.numpy()
    # x_test1 = mnist_test.data.numpy()[:20]
    x_test1 = mnist_test.data.numpy()[random_range]
    y_test1 = mnist_test.targets.numpy()[random_range]

    #%%

    x_train2 = np.array(cifar_train.data)
    y_train2 = np.array(cifar_train.targets)
    x_test2 = np.array(cifar_test.data)[:20]
    y_test2 = np.array(cifar_test.targets)[:20]


    # %%



    # %%
    model1 = Image_classifier()
    model1.train(x_train1, y_train1)

    model2 = Image_classifier()
    model2.train(x_train2, y_train2)

    #%%
    k_values = np.arange(1, 100, 10)
    # acc1 = []
    acc2 = []

    for k in k_values:
        # y_pred1 = model1.predict(x_test1, k=k)
        # accuracy1 = np.mean(y_pred1 == y_test1)
        # print(f'k={k}: Accuracy on MNIST: {accuracy1*100:.1f}%')
        # acc1.append(accuracy1)

        y_pred2 = model2.predict(x_test2, k=k)
        accuracy2 = np.mean(y_pred2 == y_test2)
        print(f'k={k}: Accuracy on CIFAR-10: {accuracy2*100:.1f}%')
        acc2.append(accuracy2)

    #%%
    fig, ax = plt.subplots(2,1, figsize=(8,10))
    ax[0].plot(np.arange(1, 51, 10), np.array(acc1)*100, marker='o')
    ax[0].set_title('MNIST Accuracy vs k-neighbors')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].grid()

    ax[1].plot(k_values, np.array(acc2)*100, marker='o', color='orange')
    ax[1].set_title('CIFAR-10 Accuracy vs k-neighbors')
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].grid()
    plt.savefig('ej2_accuracy_plots.png',dpi=300)

    # %%

    # %%
    svm_model = SVM_classifier()
    svm_model.train(x_train1, y_train1)
# %%
    y_pred1 = svm_model.predict(x_test1)
    accuracy1 = np.mean(y_pred1 == y_test1)
    print(f'SVM Accuracy on MNIST: {accuracy1*100:.1f}%')