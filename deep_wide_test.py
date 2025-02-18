import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from Layers.RKS_Layer import RKS_Layer
from Layers.FastFood_Layer import FastFood_Layer
from NN import run_NN
import pickle
import os
from collections import namedtuple
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Pad(2, padding_mode="edge")
])

# Import data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split data into batches, and shuffle
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Placeholders
rksmoduleList = nn.ModuleList()
ffmoduleList = nn.ModuleList()

dims = [1024, 4096, 8192, 16384, 16384, 16384, 16384]

for in_dim, out_dim in zip(dims, dims[1:]):
    rksmoduleList.append(RKS_Layer(input_dim=in_dim, output_dim=out_dim, scale=5, device=device, learn_G=False, nonlinearity=False))
    rksmoduleList.append(nn.BatchNorm1d(out_dim, affine=False))
    rksmoduleList.append(nn.ReLU())

    ffmoduleList.append(FastFood_Layer(input_dim=in_dim, output_dim=out_dim, scale=5, device=device, learn_G=False, nonlinearity=False))
    ffmoduleList.append(nn.BatchNorm1d(out_dim, affine=False))
    ffmoduleList.append(nn.ReLU())

rksmoduleList.append(nn.Linear(dims[-1], 10))
ffmoduleList.append(nn.Linear(dims[-1], 10))

# Train models
rks_results = run_NN(trainloader, testloader, rksmoduleList, args.epochs, device, args.lr)
ff_results = run_NN(trainloader, testloader, ffmoduleList, args.epochs, device, args.lr)

# Extract accuracy values
_, _, rks_train_acc, rks_test_acc, _, _ = rks_results
_, _, ff_train_acc, ff_test_acc, _, _ = ff_results

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs + 1), rks_train_acc, label='RKS Train Accuracy', linestyle='dashed')
plt.plot(range(1, args.epochs + 1), rks_test_acc, label='RKS Test Accuracy')
plt.plot(range(1, args.epochs + 1), ff_train_acc, label='FF Train Accuracy', linestyle='dashed')
plt.plot(range(1, args.epochs + 1), ff_test_acc, label='FF Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy for RKS and FF Models')
plt.legend()
plt.grid()
plt.show()
