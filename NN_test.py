import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import math
from Layers.RKS_Layer import RKS_Layer
from Layers.Name_Pending_Layer import BIG_Fastfood_Layer as Big_FastFood

class NeuralNetwork(nn.Module):
    def __init__(self, projections, proj_dim, output_dim, linearity=False):
        super(NeuralNetwork, self).__init__()
        self.linearity = linearity
        # Store projections as a ModuleList (separate for each group of projections)
        self.projections = nn.ModuleList(projections)
        
        # If desired, apply ReLU after each projection
        if self.linearity:
            self.relu = nn.ReLU()
        
        # Output layer
        self.output = nn.Linear(proj_dim, output_dim)

    def forward(self, x):
        # Pass through each projection in the ModuleList
        for proj in self.projections:
            x = proj(x)
            if self.linearity:
                x = self.relu(x)
        x = self.output(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Params
num_epochs = 5
scale = 10

# Loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Pad(2, padding_mode="edge")])

# Import data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split data into batches, and shuffle
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

# Projections (3 of each type initialized separately)
rks_projections = [
    RKS_Layer(input_dim=1024, output_dim=2048, scale=scale, device=device, nonlinearity=False),
    RKS_Layer(input_dim=2048, output_dim=2048, scale=scale, learn_G=True, device=device, nonlinearity=False),
    RKS_Layer(input_dim=2048, output_dim=2048, scale=scale, device=device, nonlinearity=False),
]

rks_learnable_projections = [
    RKS_Layer(input_dim=1024, output_dim=2048, scale=scale, learn_G=True, device=device, nonlinearity=False),
    RKS_Layer(input_dim=2048, output_dim=2048, scale=scale, learn_G=True, device=device, nonlinearity=False),
    RKS_Layer(input_dim=2048, output_dim=2048, scale=scale, learn_G=True, device=device, nonlinearity=False),
]

fastfood_projections = [
    Big_FastFood(input_dim=1024, output_dim=2048, scale=scale, device=device, nonlinearity=False),
    Big_FastFood(input_dim=2048, output_dim=2048, scale=scale, device=device, nonlinearity=False),
    Big_FastFood(input_dim=2048, output_dim=2048, scale=scale, device=device, nonlinearity=False),
]

fastfood_learnable_projections = [
    Big_FastFood(input_dim=1024, output_dim=2048, scale=scale, device=device, nonlinearity=False),
    Big_FastFood(input_dim=2048, output_dim=2048, scale=scale, device=device, nonlinearity=False),
    Big_FastFood(input_dim=2048, output_dim=2048, scale=scale, device=device, learn_S=True, learn_G=True, learn_B=True, nonlinearity=False),
]

name = ["RKS", "RKS_Learnable", "FastFood", "FastFood_Learnable"]

# For each projection setup (pass each list of projections separately)
# for idx, proj_list in enumerate([rks_projections, rks_learnable_projections, fastfood_projections, fastfood_learnable_projections]):
for idx, proj_list in enumerate([fastfood_learnable_projections]):
    print(f"Model with projection type: {name[idx]}")

    start = time.time()

    # Neural network with 3 hidden layers
    NN = NeuralNetwork(projections=proj_list, proj_dim=2048, output_dim=10, linearity=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(NN.parameters(), lr=0.001)

    # Train the network
    for epoch in range(num_epochs):
        NN.train()

        for images, labels in trainloader:
            # Flatten the input images to (batch_size, 784)
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = NN(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Evaluate the model
        NN.eval()
        correct = 0
        total = 0
        test_start = time.time()
        with torch.no_grad():
            for images, labels in testloader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)

                outputs = NN(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_end = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%, Completed in: {test_end-test_start:.2f} seconds")

    # Timing end
    end = time.time()
    elapsed_time = end - start
    print(f"Training completed in: {elapsed_time:.2f} seconds\n")
