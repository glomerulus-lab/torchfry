import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Layers.RKS_Layer import RKS_Layer
from Layers.FastFood_Layer import FastFood_Layer
from NN import run_NN
import pickle
import os
from collections import namedtuple
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sweep_params():
    IterationData = namedtuple("IterationData", 
        ["projection", "learnable", "learnable_gbs", "scale", "projection_dimensions", "epochs",
         "batch_size", "batch_norm", "lr"])
    epochs = 10
    batch_size = 512
    batch_norm = False
    lr = 0.1
    scales = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    projection_dimensions = [[2048, 2048, 2048]]

    # All combinations of learnable layers
    for projection in ["rks", "ff"]:
        n = 8 if projection=="ff" else 2
        for i in range(n):
            learnable = False if i == 0 else True
            learnable_gbs = [bool(int(b)) for b in f"{i:03b}"]

            for scale in scales:
                for proj_dim in projection_dimensions:
                    yield IterationData(projection, 
                                        "NA" if projection == "ff" else learnable, 
                                        "NA" if projection == "rks" else learnable_gbs, 
                                        scale, proj_dim, epochs, batch_size, batch_norm, lr)

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("--projection", type=str, choices=["ff", "rks"], help="Projection technique desired (ff or rks)", required=True)
    parser.add_argument("--learnable", type=bool, help="Learnable Projection weights (bool)", default=False)
    parser.add_argument("--learnable_gbs", type=list[bool], default=[False, False, False], help="Learnable Projection weights (list of floats)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for projection (float)")
    parser.add_argument("--projection_dimensions", type=list[int], default=[2048, 4096, 8192], help="Projection dims by layer (list of ints)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--batch_norm", help="Use batch normalization (bool)", default=False)
    parser.add_argument("--lr", help="LR of optimizer", default=0.1)

    return parser.parse_args()

for args in sweep_params():
    print(args)

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Placeholders
    moduleList = nn.ModuleList()
    input_dim = 1024
    for i in range(len(args.projection_dimensions)):
        if args.projection == 'rks':
            moduleList.append(RKS_Layer(input_dim=input_dim, 
                                        output_dim=args.projection_dimensions[i], 
                                        scale=args.scale, 
                                        device=device, 
                                        learn_G=args.learnable,
                                        nonlinearity=False))
        else:
            moduleList.append(FastFood_Layer(input_dim=input_dim, 
                                        output_dim=args.projection_dimensions[i], 
                                        scale=args.scale, 
                                        device=device, 
                                        learn_G=args.learnable_gbs[0],
                                        learn_B=args.learnable_gbs[1],
                                        learn_S=args.learnable_gbs[2],
                                        nonlinearity=False))
        # Batch Norm
        if args.batch_norm:
            moduleList.append(nn.BatchNorm1d(affine=False))
        # Nonlinearity
        moduleList.append(nn.ReLU())
        input_dim = args.projection_dimensions[i]

    # Output Layer
    moduleList.append(nn.Linear(input_dim, 10))

    learnable_params, non_learnable_params, train_accuracy, test_accuracy, elapsed_time, test_time = run_NN(trainloader, testloader, moduleList, args.epochs, device, args.lr)

    os.makedirs("testing_performance", exist_ok=True)

    # Define the filename
    filename = f'testing_performance/proj={args.projection}-learn={args.learnable}-learn_gbs={args.learnable_gbs}-scale={args.scale}-projdims={args.projection_dimensions}-epoch={args.epochs}-batch_size={args.batch_size}-batch_norm={args.batch_norm}.pkl'

    # Define a dictionary to store the hyperparameters and performance metrics
    hyperparams_and_performance = {
        "hyperparameters": {
            "projection": args.projection,
            "learnable": args.learnable,
            "learnable_gbs": args.learnable_gbs,
            "scale": args.scale,
            "projection_dimensions": args.projection_dimensions,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "batch_norm": args.batch_norm
        },
        "performance": {
            "learnable_params": learnable_params,
            "non_learnable_params": non_learnable_params,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "elapsed_time": elapsed_time,
            "test_time": test_time
        }
    }

    # Save the dictionary to a file
    with open(filename, 'wb') as f:
        pickle.dump(hyperparams_and_performance, f)

    print(f"Hyperparameters and performance saved to {filename}")
