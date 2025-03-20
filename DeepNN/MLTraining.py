import DataReading as DR
import torch 
import numpy as np
import matplotlib.pyplot as plt
from NNParameters import model, loss_fn, n_hidden_neurons, learning_rate
X, t = DR.readData("train")

#Define Parameters
n_epochs = 1000

#Normalise data 

#Define input to torch
X = torch.tensor(X, dtype=torch.float)
t = torch.tensor(t, dtype=torch.float).view(-1,1)

#10 hidden layers

optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(n_epochs):
    y_pred = model(X)

    loss = loss_fn(y_pred, t)

    #Backpropagation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    print(i)

torch.save(model.state_dict(), "Trained_MLModel.pth")
