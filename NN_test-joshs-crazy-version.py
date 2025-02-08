import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
import pandas as pd
import math
from collections import namedtuple
from Layers.RKS_Layer import RKS_Layer
from Layers.Name_Pending_Layer import BIG_Fastfood_Layer as Name_Pending_Layer
import wrangle_data
import argparse

class Neural_Network(nn.Module):
    def __init__(self, dimensions, scale, layer_type, learnables, nonlinearity, device):
        super(Neural_Network, self).__init__()

        layers = []
        for i in range(len(dimensions)-1):

            if layer_type == "Name_Pending":
                layer = Name_Pending_Layer(
                    dimensions[i], dimensions[i+1], scale, 
                    learn_B= learnables[0], learn_G= learnables[1], learn_S= learnables[2],
                    nonlinearity= nonlinearity=="cos", device=device
                )
                layers.append(layer)

            elif layer_type == "RKS":
                layer = RKS_Layer(
                    dimensions[i], dimensions[i+1], scale,
                    learn_G= learnables[0],
                    nonlinearity= nonlinearity=="cos", device=device
                )
                layers.append(layer)

            else:
                raise ValueError(f"Impossible argument type. {layer_type} is not a valid layer.")

            if nonlinearity == "relu" and i != len(dimensions)-1:
                # relu nonlinearity and not the last layer in the network
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def iterate():
    IterationData = namedtuple("IterationData", 
        ["layer_type", "scale", "epochs", "depth", "batch_size", "proj_dim", "nonlinearity", "dataset"])
    epochs = 10
    batch_size = 512
    network_depth = [3,2,1]
    scales = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    projection_dimensions = [1024, 2048, 4096]
    nonlinearites = ["none", "relu", "cos"]
    datasets = ["red_wine"]
    layer_types = [
        ("Name_Pending", np.array([0,0,0], dtype=bool)),
        ("Name_Pending", np.array([0,0,1], dtype=bool)),
        ("Name_Pending", np.array([0,1,0], dtype=bool)),
        ("Name_Pending", np.array([0,1,1], dtype=bool)),
        ("Name_Pending", np.array([1,0,0], dtype=bool)),
        ("Name_Pending", np.array([1,0,1], dtype=bool)),
        ("Name_Pending", np.array([1,1,0], dtype=bool)),
        ("Name_Pending", np.array([1,1,1], dtype=bool)),
        ("RKS", np.array([0], dtype=bool)),
        ("RKS", np.array([1], dtype=bool)),
    ]
    for dataset in datasets:
        for proj_dim in projection_dimensions:
            for depth in network_depth:
                for scale in scales:
                    for nonlinearity in nonlinearites:
                        for layer_type in layer_types:
                            yield IterationData(layer_type, scale, epochs, depth, batch_size, proj_dim, nonlinearity, dataset)

data_list = {
    "iris": wrangle_data.load_iris,
    "animal_center": wrangle_data.load_animal_center,
    "parkinsons": wrangle_data.load_parkinsons,
    "red_wine": wrangle_data.load_red_wine_quality,
    "white_wine": wrangle_data.load_white_wine_quality,
    "insurance": wrangle_data.load_insurance,
    "CT_slices": wrangle_data.load_CT_slices,
    # "KEGG_network": wrangle_data.load_KEGG_network, # TODO: missing target variable
    "year_prediction_MSD": wrangle_data.load_year_prediction_MSD,
}

parser = argparse.ArgumentParser(description="Run the script with verbose output")
parser.add_argument('--verbose', '--v', action='store_true', help="Enable verbose mode")
args = parser.parse_args()

columns=["layer_type", "scale", "epochs", "depth", "batch_size", "proj_dim", "nonlinearity", 
         "dataset", "learning_time", "train_forward_time", "test_forward_time", "train_loss", "test_loss"]
log_results = pd.DataFrame(columns=columns)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For each method
for name in iterate():

    xtrain, ytrain, xtest, ytest, info = data_list[name.dataset]()
    input_dim = info.input_dim
    output_dim = info.output_dim

    dimensions = [input_dim, *([name.proj_dim] * (name.depth-1)), output_dim]

    NN = Neural_Network(dimensions, name.scale, name.layer_type[0], name.layer_type[1], name.nonlinearity, device)
    criterion = nn.CrossEntropyLoss() if info.is_categorical else nn.MSELoss()
    optimizer = optim.Adam(NN.parameters(), lr=0.001) # TODO: NN.parameters() is nothing

    start = time.time()
    for epoch in range(name.epochs):
        NN.train()

        # sample a batch of data
        batch_idx = [torch.randperm(xtrain.shape[0])[:name.batch_size]]
        x_batch = xtrain[batch_idx]
        y_batch = ytrain[batch_idx]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = NN(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        NN.eval()

    learning_time = time.time() - start

    # Find accuracy on test & training sets
    with torch.no_grad():

        # Find loss for categorical and regression data separetely
        start = time.time()
        outputs = NN(xtrain)
        train_forward_time = time.time() - start
        if info.is_categorical:
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == ytrain).sum().item()
            total = ytrain.size(0)
            train_loss = 100 * correct / total
        else:
            train_loss = criterion(outputs, ytrain)

        # Find loss for categorical and regression data separetely
        start = time.time()
        outputs = NN(xtest)
        test_forward_time = time.time() - start
        if info.is_categorical:
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == ytest).sum().item()
            total = ytest.size(0)
            test_loss = 100 * correct / total
        else:
            test_loss = criterion(outputs, ytest)

    new_row = pd.Series(name + (learning_time, train_forward_time, test_forward_time, train_loss, test_loss), index=columns)
    log_results = log_results.append(new_row, ignore_index=True)
