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
num_epochs = 20
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

# Projections
# rks = RKS_Layer(input_dim=784, output_dim=1024, scale=scale, device=device)
# rks_learn = RKS_Layer(input_dim=784, output_dim=1024, scale=scale, learn_G=True, device=device)
ff = Big_FastFood(input_dim=1024, output_dim=2048, scale=scale, device=device)
ff_learn = Big_FastFood(input_dim=1024, output_dim=2048, scale=scale, device=device, learn_S=True, learn_G=True, learn_B=True)
ff_no_linear = Big_FastFood(input_dim=1024, output_dim=2048, scale=scale, device=device, nonlinearity=False)
projs = [ff, ff_learn, ff_no_linear]

# For each projection
for proj in projs:
    start = time.time()
    NN = NeuralNetwork(projection=proj, proj_dim=2048, output_dim=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(NN.parameters(), lr=0.001)

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

        NN.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)

                outputs = NN(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")

    # Timing end
    end = time.time()
    elapsed_time = end - start
    print(f"Training completed in: {elapsed_time:.2f} seconds\n")