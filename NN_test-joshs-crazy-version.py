import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import math
from Layers.RKS_Layer import RKS_Layer
from Layers.Name_Pending_Layer import BIG_Fastfood_Layer as Big_FastFood

class NeuralNetwork(nn.Module):
    def __init__(self, projection, proj_dim, output_dim, linearity=False):
        super(NeuralNetwork, self).__init__()
        # If desired nonlinearity
        self.linearity = linearity

        # Single hidden layer, with relu if desired(Instead of internal projection nonlinearity)
        self.projection = projection
        if self.linearity:
            self.relu = nn.ReLU()
        self.output = nn.Linear(proj_dim, output_dim)
        

    def forward(self, x):
        x = self.projection(x)
        if self.linearity:
            x = self.relu(x)
        x = self.output(x)
        return x
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Params
num_epochs = 500
scale = 10

def next_power_of_two(x):
    return 2**math.ceil(math.log2(x))

class PadToNextPowerOfTwo(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        # Get the original size
        width, height = image.size
        
        # Find the next power of two for both dimensions
        new_width = next_power_of_two(width)
        new_height = next_power_of_two(height)
        
        # Calculate the padding for both sides
        pad_left = (new_width - width) // 2
        pad_top = (new_height - height) // 2
        pad_right = new_width - width - pad_left
        pad_bottom = new_height - height - pad_top
        
        # Apply padding
        return transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

# Loader
transform = transforms.Compose([PadToNextPowerOfTwo(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Import data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Split data into batches, and shuffle
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

from wrangle_data import load_insurance
xtrain, ytrain, xtest, ytest = load_insurance()

# Projections
input_dim = xtrain.shape[1]
projection_dim = 1028
projs = [
    RKS_Layer(input_dim, projection_dim, scale=scale, device=device, nonlinearity=False),
    RKS_Layer(input_dim, projection_dim, scale=scale, learn_G=True, device=device, nonlinearity=False),
    Big_FastFood(input_dim, projection_dim, scale=scale, device=device, nonlinearity=False),
    Big_FastFood(input_dim, projection_dim, scale=scale, device=device, learn_S=True, learn_G=True, learn_B=True, nonlinearity=False),
]
name = ["RKS", "RKS_Learnable", "FastFood", "FastFood_Learnable"]
# For each projection
for i in range(len(projs)):
    print(name[i])
    start = time.time()
    NN = NeuralNetwork(projection=projs[i], proj_dim=projection_dim, output_dim=10, linearity=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(NN.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        NN.train()

        # sample a batch of data
        batch_idx = [torch.randperm(xtrain.shape[0])[:512]]
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
        correct = 0
        total = 0

        with torch.no_grad():
            
            # sample a batch of data
            batch_idx = [torch.randperm(xtest.shape[0])[:512]]
            x_batch = xtest[batch_idx]
            y_batch = ytest[batch_idx]

            outputs = NN(x_batch)
            # if regression:
            #     loss = criterion(outputs, y_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            
            correct += (predicted == y_batch).sum().item()
            
            accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")

    # Timing end
    end = time.time()
    elapsed_time = end - start
    print(f"Training completed in: {elapsed_time:.2f} seconds\n")