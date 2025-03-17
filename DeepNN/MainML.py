import DataReading as DR
import torch 
import numpy as np
import matplotlib.pyplot as plt
X, t = DR.readData("train")

#Define Parameters
n_hidden_neurons = 40
n_epochs = 100

#Normalise data 


#Define input to torch
X = torch.tensor(X, dtype=torch.float)
t = torch.tensor(t, dtype=torch.float).view(-1,1)

#10 hidden layers
model = torch.nn.Sequential(torch.nn.Linear(2, n_hidden_neurons), torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,n_hidden_neurons), torch.nn.ReLU(), torch.nn.Linear(n_hidden_neurons,1))
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-9
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(n_epochs):
    y_pred = model(X)

    loss = loss_fn(y_pred, t)

    #Backpropagation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    print(y_pred)


#Use the test data




