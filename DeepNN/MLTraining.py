import DataReading as DR
import torch 
import numpy as np
import matplotlib.pyplot as plt
from NNParameters import model, loss_fn, n_epochs, learning_rate
from sklearn.preprocessing import StandardScaler

#Check if cuda device available 
computationDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, t = DR.readData2("train")

#Normalise data 
scaler = StandardScaler()
X = scaler.fit_transform(X)
t = scaler.fit_transform(np.array(t).reshape(-1,1)).flatten()

#Define input to torch
X = torch.tensor(X, dtype=torch.float)
t = torch.tensor(t, dtype=torch.float).view(-1,1)

#Initialise cuda
X = X.to(device=computationDevice)
t = t.to(device=computationDevice)
model = model.to(device=computationDevice)


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
