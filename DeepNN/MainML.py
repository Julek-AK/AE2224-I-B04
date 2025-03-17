import DataReading as DR
import torch 
import numpy as np
import matplotlib.pyplot as plt
X, t = DR.readData("train")

#Define Parameters
n_hidden_neurons = 10
n_epochs = 5000

#Define input to torch
X = torch.tensor(X, dtype=torch.float)
t = torch.tensor(t, dtype=torch.float).view(-1,1)

model = torch.nn.Sequential(torch.nn.Linear(2, n_hidden_neurons), torch.nn.Sigmoid(), torch.nn.Linear(n_hidden_neurons,1))
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-5
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(n_epochs):
    y_pred = model(X)

    loss = loss_fn(y_pred, t)

    #Backpropagation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


#Use the test data

X_test, t_test = DR.readData("test")
X_test = torch.tensor(X_test, dtype=torch.float)
t_test = torch.tensor(t_test, dtype=torch.float).view(-1,1)

y_pred = model(X_test)


print(y_pred)


