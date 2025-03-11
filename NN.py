import torch
import torch.nn as nn
import torch.optim as optim
import time

class NeuralNetwork(nn.Module):
    def __init__(self, projections):
        super(NeuralNetwork, self).__init__()
        # Store projections as a ModuleList (separate for each group of projections)
        self.projections = projections

    def forward(self, x):
        # Pass through each projection in the ModuleList
        for proj in self.projections:
            x = proj(x)
        return x

# Learnable and Non-Learnable Params
def count_params(model):
    learnable_params = 0
    non_learnable_params = 0

    for param in model.parameters():
        if param.requires_grad:
            learnable_params += param.numel()
        else:
            non_learnable_params += param.numel()
    
    return learnable_params, non_learnable_params


def run_NN(trainloader, testloader, layers: nn.ModuleList, epochs, device, lr):

    start_time = time.time()
    train_accuracy, test_accuracy = [], []
    train_times, forward_pass_times = [], []

    for layer in layers:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    model = NeuralNetwork(layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # Train the network
    for epoch in range(epochs):
        epoch_time = time.time()
        model.train()

        for images, labels in trainloader:
            # Flatten the input images to (batch_size, 784)
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        train_times.append(time.time() - epoch_time)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            # Train accuracy 
            correct, total = 0, 0
            for images, labels in trainloader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_accuracy.append(100 * correct / total)

            # Test accuracy & time
            correct, total = 0, 0
            testing_time_start = time.time()
            for images, labels in testloader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)

                # Time the model forward pass
                test_start = time.time()
                outputs = model(images)
                forward_pass_times.append(time.time() - test_start)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_accuracy.append(100 * correct / total)
            test_time = time.time() - testing_time_start
        print(f"Epoch [{epoch+1}/{epochs}], Test Accuracy: {test_accuracy[-1]:.2f}%, Forward pass time: {test_time:.2f} seconds")


    # Get learnable and non-learnable parameters from model
    learnable_params, non_learnable_params = count_params(model)

    # Timing of training across all epochs
    elapsed_time = time.time() - start_time
    print(f"Training completed in: {elapsed_time:.2f} seconds\n")

    return learnable_params, non_learnable_params, train_accuracy, test_accuracy, elapsed_time, train_times, forward_pass_times

    

