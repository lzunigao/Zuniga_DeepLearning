#%%
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
#%%
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, lr=1e-3, reg=1e-4):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.lr = lr
        self.reg = reg
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.linear(x)

    def loss_fn(self, scores, y):
        raise NotImplementedError("Subclasses must implement loss_fn")

    def fit(self, train_loader, val_loader=None, epochs=10, device='cpu'):
        self.to(device)
        
        train_acc = []
        train_loss = []

        val_acc = []

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for X, y in train_loader:
                X, y = X.view(X.size(0), -1).to(device), y.to(device)
                scores = self.forward(X)
                loss = self.loss_fn(scores, y)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    preds = self.predict(X)
                    total_correct += (preds == y).sum().item()
                    total_samples += y.size(0)

            acc = total_correct / total_samples
            if val_loader:
                vacc = self.evaluate(val_loader, device)
                
                print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, train_acc={acc:.4f}, val_acc={vacc:.4f}")
                train_acc.append(acc)
                train_loss.append(total_loss/len(train_loader))
                val_acc.append(vacc)
                
                
            else:
                print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, train_acc={acc:.4f}")
                train_acc.append(acc)
                train_loss.append(total_loss/len(train_loader))

        return train_acc, train_loss, val_acc

    def predict(self, X):
        scores = self.forward(X)
        return torch.argmax(scores, dim=1)
    
    def evaluate(self, loader, device='cpu'):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.view(X.size(0), -1).to(device), y.to(device)
                preds = self.predict(X)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

# %%

class LinearSVM(LinearClassifier):
    def __init__(self, input_dim, num_classes, lr=1e-3, reg=1e-4):
        super().__init__(input_dim, num_classes, lr, reg)
        self.hinge = nn.MultiMarginLoss()  # built-in multiclass hinge loss

    def loss_fn(self, scores, y):
        # Compute hinge loss
        loss = self.hinge(scores, y)

        # Add L2 regularization
        l2 = 0
        for param in self.parameters():
            l2 += torch.sum(param ** 2)
        return loss + 0.5 * self.reg * l2
    
#%%
    
class SoftmaxClassifier(LinearClassifier):
    def loss_fn(self, scores, y):
        ce_loss = nn.functional.cross_entropy(scores, y)
        l2 = 0
        for param in self.parameters():
            l2 += torch.sum(param ** 2)
        return ce_loss + 0.5 * self.reg * l2

#%%

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=False)


train_data, val_data = random_split(train_data, [50000, 10000])

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256)
test_loader = DataLoader(test_data, batch_size=256)


# %%
device = 'cpu'
input_dim = 28 * 28
num_classes = 10

print("Training Linear SVM...")
svm_model = LinearSVM(input_dim, num_classes)
results_svm = svm_model.fit(train_loader, val_loader, epochs=10, device=device)
test_acc_svm = svm_model.evaluate(test_loader, device)

# %%
train_acc, train_loss, val_acc, val_loss = results_svm


fig,ax = plt.subplots(1,2, figsize=(8,4))
ax[0].plot(train_loss, label='Training Loss', ls='-.')
# ax[0].plot(val_loss, label='Validation Loss', ls='-.')
ax[0].set_title('SVM - Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].grid()

ax[1].plot(train_acc, label='Train Accuracy', ls='-.')
ax[1].plot(val_acc, label='Validation Accuracy', ls='-.')
ax[1].set_title('SVM - Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

# %%

print("\nTraining Softmax Classifier...")
input_dim = 3 * 32 * 32  # 3072 features per image
num_classes = 10
softmax_model = SoftmaxClassifier(input_dim, num_classes, lr=1e-3, reg=1e-4)
results_softmax = softmax_model.fit(train_loader, val_loader, epochs=10, device=device)
test_acc_softmax = softmax_model.evaluate(test_loader, device)
#%%
train_acc, train_loss, val_acc = results_softmax

fig,ax = plt.subplots(1,2, figsize=(8,4))
ax[0].plot(train_loss, label='Train Loss', ls='-.')
# ax[0].plot(val_loss, label='Validation Loss', ls='-.')
ax[0].set_title('Softmax - Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].grid()

ax[1].plot(train_acc, label='Train Accuracy', ls='-.')
ax[1].plot(val_acc, label='Validation Accuracy', ls='-.')
ax[1].set_title('Softmax - Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

# %% CIFAR

train_data_cifar = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_data_cifar = datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)


train_data_cifar, val_data_cifar = random_split(train_data_cifar, [40000, 10000])

train_loader = DataLoader(train_data_cifar, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data_cifar, batch_size=256)
test_loader = DataLoader(test_data_cifar, batch_size=256)



print("Training Linear SVM...")
svm_model = LinearSVM(input_dim, num_classes)
results_svm = svm_model.fit(train_loader, val_loader, epochs=10, device=device)
test_acc_svm = svm_model.evaluate(test_loader, device)


softmax_model = SoftmaxClassifier(input_dim, num_classes, lr=1e-3, reg=1e-4)
results_softmax = softmax_model.fit(train_loader, val_loader, epochs=10, device=device)
test_acc_softmax = softmax_model.evaluate(test_loader, device)
# %%
