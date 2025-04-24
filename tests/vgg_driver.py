import json
import torch
import argparse
import os
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time
from fastfood_torch.networks import VGG
from fastfood_torch.transforms import FastFoodLayer, RKSLayer

# Determine the device to be used for computation (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_params(model):
    """
    Count the number of learnable and non-learnable parameters in a model.
    Args:
        model (torch.nn.Module): The model to analyze.
    Returns:
        tuple: A tuple containing the number of learnable parameters and non-learnable parameters.
    """
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return learnable_params, non_learnable_params

def parse_all_args():
    """
    Parse command-line arguments for the script.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run experiments based on configurations in .json file")
    parser.add_argument('--config', type=str, help="Desired config to run (ex: configs.json)")
    parser.add_argument('--filename', type=str, help="Filename for saving results of the run (ex: results.json)")
    return parser.parse_args()


# Mapping of layer names to their corresponding classes 
layer_map = {
    "FastFoodLayer": FastFoodLayer,
    "RKSLayer": RKSLayer
}

# Parse command-line arguments
args = parse_all_args()

# Load configuration parameters from the JSON file
with open(args.config, "r") as f:
    sweep = json.load(f)

# Initialize a list to store all experiment results
all_results = []

# Iterate over each configuration in the sweep
for config in sweep:
    print(config)
    
    # Extract the layer name and retrieve the corresponding class
    layer_name = config.pop("layer")
    projection = layer_map[layer_name]
    config["device"] = str(device)

    # Backup the original config before popping
    original_config = config.copy()

    # Define data transformations and load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["mb"], shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["mb"], shuffle=False)

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
    for _ in range(trials):
        print(f"Trial {_}:")
        
        # Initialize the model with specified parameters
        model = VGG(
            projection_layer=projection,
            input_shape=(3, 224, 224),
            features=features,
            classes=10,
            proj_args=config)
        model.to(device)

        # Initialize lists to store metrics for each epoch
        train_accuracy, test_accuracy = [], []
        train_times, forward_pass_times = [], []

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)


        # Record the start time of the training process
        start_time = time.time()

        # Training loop over the specified number of epochs
        for epoch in range(epochs):
            epoch_time = time.time()
            model.train()
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Record the time taken for the current epoch
            train_times.append(time.time() - epoch_time)
            
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                # Calculate training accuracy
                correct, total = 0, 0
                for images, labels in trainloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                train_accuracy.append(100 * correct / total)

                # Calculate test accuracy and average forward pass time
                correct, total = 0, 0
                forward_pass_time = 0
                for images, labels in testloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    test_start = time.time()
                    outputs = model(images)
                    forward_pass_time += time.time() - test_start
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                test_accuracy.append(100 * correct / total)

                
                forward_pass_times.append(forward_pass_time / len(testloader))
                print(f"Epoch {epoch}: Train Accuracy = {train_accuracy[epoch]:.2f}%, Test Accuracy = {test_accuracy[epoch]:.2f}%")

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

    # Append the results of the current configuration to the overall results list
    all_results.append(results)
    
# Check for dir existence
if not os.path.exists("results"):
    os.makedirs("results")

# Save all experiment results to a JSON file
with open(os.path.join("results", f"{args.filename}"), "w") as f:
    json.dump(all_results, f, indent=4)

print(f"Runs complete, saved to {args.filename}")
