import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
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
num_epochs = 100
scale = 10

# Loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Import data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Split data into batches, and shuffle
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# Projections
rks = RKS_Layer(input_dim=784, output_dim=2048, scale=scale, device=device)
rks_learn = RKS_Layer(input_dim=784, output_dim=2048, scale=scale, learn_G=True, device=device)
ff = Big_FastFood(input_dim=784, output_dim=2048, scale=scale, device=device)
ff_learn = Big_FastFood(input_dim=784, output_dim=2048, scale=scale, device=device, learn_S=True, learn_G=True, learn_B=True)
ff_no_linear = Big_FastFood(input_dim=784, output_dim=2048, scale=scale, device=device, nonlinearity=False)
projs = [rks, rks_learn, ff, ff_learn, ff_no_linear]

# For each projection
for proj in projs:
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