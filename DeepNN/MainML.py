import DataReading as DR
import torch 
import numpy as np
X, t = DR.readData()

#Define Parameters
n_hidden_neurons = 10
n_epochs = 5000

#Define input to torch
X = torch.tensor(X)
t = torch.tensor(t)

model = torch.nn.Sequential(torch.nn.Linear(2, n_hidden_neurons), torch.nn.Sigmoid(), torch.nn.Linear(n_hidden_neurons, 1))
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-5
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(n_epochs):
    y_pred = model(X)

