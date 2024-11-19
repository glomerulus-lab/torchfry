
'''
Creates an MLP to fit to the iris dataset
With the Fastfood Layer supplied as a Non-Linearity
Tested for performance against a ReLU activation function
'''

import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch import nn
import matplotlib.pyplot as plt


class LinearMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        '''
        Returns -- raw outputs of network
                    class prediciton
        '''
        logits = self.network(x)
        return self.softmax(logits)

class FastfoodMLP(nn.Module):
    '''Uses the fastfood layer for non-linearity'''
    def __init__(self, input_dim, fastfood_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.fastfood_dim = fastfood_dim
        self.output_dim = output_dim

        import os
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from FastFood_Layer import Fastfood_Layer

        self.network = nn.Sequential(
            Fastfood_Layer(self.input_dim, self.fastfood_dim, scale=1),
            nn.Linear(self.fastfood_dim, self.output_dim),
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        logits = self.network(x)
        return self.softmax(logits)
    

def train_model(model, features, targets, epochs=1000, lr=0.01):
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(epochs):
        y_pred = model.forward(features)
        loss = criterion(y_pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses
    
def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_test).float().mean()
    return accuracy.item()

def plot(model, losses, x_train, y_train, x_test, y_test):
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')

    # Calculate and display accuracy
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    ypred_on_train = torch.argmax(model.forward(x_train), dim=1)
    ypred_on_test = torch.argmax(model.forward(x_test), dim=1)

    # Plot predictions vs actual for first two features
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        predictions = model(x_test).argmax(dim=1)
        plt.scatter(y_train, ypred_on_train, c=y_train, cmap='viridis', alpha=0.6, label='Training Data')
        plt.scatter(y_test, ypred_on_test, c=predictions, cmap='viridis', marker='x', alpha=0.6, label='Predictions')
        plt.xlabel('Actual y')
        plt.ylabel('Predicted y')
        plt.title('Predictions vs Actual')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def bootstrap(X, y, p=.5):
    """

    :param X:
    :param y:
    :return: X, y (p percent of the data randomly sampled)
    """
    # TODO: this is slow
    N = X.shape[0]
    indices = np.arange(0, N)
    train_idx = np.random.choice(indices, int(p*N), replace=True)
    test_idx = np.setdiff1d(indices, train_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    

if __name__ == '__main__':
    data = pd.read_csv("data\\iris.csv")
    X = data.iloc[:, :-1].to_numpy()
    y = pd.factorize(data.iloc[:,-1])[0]

    n = 10
    linear_acc = np.zeros((n,2))
    fastfood_acc = np.zeros((n,2))
    # train on n bootstrapping samples
    for idx in range(n):
        x_train, y_train, x_test, y_test = bootstrap(X, y, p=.8)
        # change from numpy arrays to torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

        model1 = LinearMLP(input_dim=x_train.shape[1], hidden_dim=256, output_dim=3)
        losses = train_model(model1, x_train, y_train)
        linear_acc[idx,0] = evaluate_model(model1, x_train, y_train)
        linear_acc[idx,1] = evaluate_model(model1, x_test, y_test)

        model2 = FastfoodMLP(x_train.shape[1], 16, 3)
        losses = train_model(model2, x_train, y_train)
        fastfood_acc[idx,0] = evaluate_model(model2, x_train, y_train)
        fastfood_acc[idx,1] = evaluate_model(model2, x_test, y_test)

    print(f"Average linear train acc: {np.average(linear_acc[:,0])}")
    print(f"Average linear test acc: {np.average(linear_acc[:,1])}")
    print(f"Average fastfood train acc: {np.average(fastfood_acc[:,0])}")
    print(f"Average fastfood test acc: {np.average(fastfood_acc[:,1])}")




        


