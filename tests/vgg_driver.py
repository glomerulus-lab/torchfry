"""
VGG Training Script

This script runs experiments for training VGG models on CIFAR-10 using FastFood or RKS projection layers.
It loads configurations from a JSON file, runs trials according to these configs, and saves results.

The script supports two projection layers:
- FastFoodLayer: Implements the FastFood transform for parameter reduction
- RKSLayer: Implements Random Kitchen Sinks for parameter reduction

Results are saved as a JSON file in the Results directory, including accuracy metrics,
training times, and parameter counts.
"""  

import json
import torch
import argparse
import os
from torchvision import datasets, transforms
import torch.nn as nn
import time
from torchfry.networks import VGG
from torchfry.transforms import FastFoodLayer, RKSLayer

# Device for operations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_params(model):
    """
    Count the number of learnable and non-learnable parameters in a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to analyze parameters
        
    Returns
    -------
    tuple of ints
        (learnable_params, non_learnable_params)
        - learnable_params: Number of parameters that require gradients
        - non_learnable_params: Number of parameters that don't require gradients
    """
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return learnable_params, non_learnable_params

def parse_all_args():
    """
    Parse command-line arguments for experiment configuration.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with the following attributes:
        - config: Path to the JSON config file to run (e.g., 'configs/vgg_exp001.json')
        - save_name: Base name for saving the result file (e.g., 'vgg_exp001_results')
    """
    parser = argparse.ArgumentParser(description="Run training using config JSON")
    parser.add_argument('--config', type=str, required=True, help="Path to the config JSON file")
    parser.add_argument('--save', type=str, required=True, help="JSON file name to save results")
    return parser.parse_args()

# Mapping of layer names to their corresponding classes
layer_map = {
    "FastFoodLayer": FastFoodLayer,
    "RKSLayer": RKSLayer
}

# Parse command-line arguments
args = parse_all_args()

# Load the specified configuration from the JSON file
with open(args.config, "r") as f:
    config = json.load(f)

# Extract the layer name and retrieve the corresponding class
layer_name = config.pop("layer")
projection = layer_map[layer_name]
config["device"] = str(device)

# Store the original configuration for later reference
original_config = config.copy()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trainloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=config["mb"], shuffle=True,
    num_workers=2, pin_memory=True)

testloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=config["mb"], shuffle=False,
    num_workers=2, pin_memory=True)

# Initialize a dictionary to store results for the current configuration
results = {
    "Projection Layer": layer_name,
    "Projection Arguments": original_config,
    "Train Accuracies": [],
    "Test Accuracies": [],
    "Elapsed Time": [],
    "Train Times Per Epoch": [],
    "Forward Pass Times": [],
    "Learnable Params": 0,
    "Non-Learnable Params": 0,
}

# Extract and remove arguments that shouldn't be passed into the model
lr = config.pop("lr")
mb = config.pop("mb")
trials = config.pop("trials")
epochs = config.pop("epochs")
features = config.pop("features")

# Conduct multiple trials for the current configuration
for trial in range(trials):
    print(f"Trial {trial}:")

    # Initialize the model with specified parameters
    model = VGG(
        projection_layer=projection,
        features=features,
        num_classes=10,
        proj_args=config
    )
    model.to(device)

    # Initialize lists to store metrics for each epoch
    train_accuracy, test_accuracy = [], []
    train_times, forward_pass_times = [], []

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=0.1,
                                                           patience=3,
                                                           threshold=0.001
                                                        )

    # Record the start time of the training process
    start_time = time.time()

    # Training loop over the specified number of epochs
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Train phase
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for batch, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y)

        train_loss /= len(trainloader)
        train_acc /= len(trainloader)

        # Evaluation phase
        model.eval()
        test_loss, test_acc = 0.0, 0.0

        with torch.inference_mode():
            for batch, (X, y) in enumerate(testloader):
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                test_loss += loss.item()

                y_pred_class = torch.argmax(y_pred, dim=1)
                test_acc += (y_pred_class == y).sum().item() / len(y)

        test_loss /= len(testloader)
        test_acc /= len(testloader)
        scheduler.step(test_acc)
        
        # Record metrics for the current epoch
        epoch_time = time.time() - epoch_start_time
        forward_pass_times.append(epoch_time / len(testloader))
        train_times.append(epoch_time)
        train_accuracy.append(100 * train_acc)
        test_accuracy.append(100 * test_acc)

        print(f"Epoch {epoch}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy[-1]:.2f}%, "
              f"Test Loss = {test_loss:.4f}, Test Acc = {test_accuracy[-1]:.2f}%")

    # Calculate the total elapsed time for the current trial
    elapsed_time = time.time() - start_time

    # Count the number of learnable and non-learnable parameters in the model
    learnable_params, non_learnable_params = count_params(model)

    # Store the results of the current trial
    results["Train Accuracies"].append(train_accuracy)
    results["Test Accuracies"].append(test_accuracy)
    results["Elapsed Time"].append(elapsed_time)
    results["Train Times Per Epoch"].append(train_times)
    results["Forward Pass Times"].append(forward_pass_times)
    results["Learnable Params"] = learnable_params
    results["Non-Learnable Params"] = non_learnable_params

# Check for dir existence
if not os.path.exists("results"):
    os.makedirs("results")

# Save all experiment results to a JSON file
result_path = os.path.join("results", args.save)
with open(result_path, "w") as f:
    json.dump([results], f, indent=4)

print(f"Run complete, saved to results/{args.save}")
