#%%
import os
import argparse
from torchvision import datasets, transforms

#%%
cifar10=datasets.CIFAR10(root='./datasets', train=True, download=False)

print(len(cifar10))
# %%
