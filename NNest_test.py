import torch
from FastFood_Layer import FastFood_Layer
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fastfood = FastFood_Layer(input_dim=4*4, output_dim = 32, scale=5, learn_G=True, learn_S=True)
        self.output = nn.Linear(32, 10)
        

    def forward(self, x):
        x = self.fastfood(x)
        x = self.output(x)
        return x
    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n = 1000
X = torch.rand(n, 4*4, device=device)
y = torch.randint(1, 10, (n,))
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()  # Or any other appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):

    optimizer.zero_grad()
    logits = model.forward(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)

    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    # print(model.fastfood.stack[0].G)
    # print(model.fastfood.stack[0].B)
    # print(model.fastfood.stack[0].S)
    print(f"Epoch = {epoch}")
    print(f"Loss = {loss}")