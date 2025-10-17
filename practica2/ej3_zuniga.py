#%%
# !export http_proxy="http://proxy.cnea.gob.ar:1280"
# !export https_proxy="http://proxy.cnea.gob.ar:1280"
# !export HTTP_proxy="http://proxy.cnea.gob.ar:1280"
# !export HTTPS_proxy="http://proxy.cnea.gob.ar:1280"
#%%
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
# %%
class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.fc_in = nn.Linear(config['input_size'], 100)
        self.fc_out = nn.Linear(100, config['num_classes']) if config['num_classes'] > 0 else nn.Linear(100, 1)
        
        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc_in(x)
        x = F.sigmoid(x)        
        x = self.fc_out(x)   

        logits = x
        probs = self.sm(logits)
        
        preds = probs.argmax(dim=1)
        preds = [self.config['id2label'][p.item()] for p in preds] if self.config['num_classes'] > 0 else preds.item()     
        return preds


# %%
BASE_PATH = Path(__file__).parent.parent
mini_batch = 2

train_ds = datasets.CIFAR10(root='./Documentos/laura/datasets/', train=True, download=False,
                           transform=transforms.Compose([transforms.ToTensor()]))
test_ds = datasets.CIFAR10(root='./Documentos/laura/datasets/', train=False, download=False,
                           transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_ds, batch_size=mini_batch, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=mini_batch, shuffle=False)


# %%
id2label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                     5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
label2id = {v: k for k, v in id2label.items()}

config = {
    'input_size': 3072,
    'num_classes': 10,
    'id2label': id2label,
    'label2id': label2id   
}

model = Net(config)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# %% LOOP DE ENTRENAMIENTO

num_epochs = 5
acc = []
loss = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate (train_loader):
        inputs = data[0].view(-1, 3072)
        labels = [id2label[d.item()] for d in data[1]] if config['num_classes'] > 0 else None
        
        outputs = model(inputs)
        # print(labels)
        # print('\nOutputs:')
        # print(outputs)
        running_acc += (np.array((outputs == labels)).sum()/len(labels)).item()
        running_loss += loss_fn (outputs, data[1])
        break
    acc.append(running_acc / len(train_loader))
    loss.append(running_loss / len(train_loader))
    print (f'Epoch {epoch+1}, Loss: {loss}')
    break
                                                        
    



# %%
