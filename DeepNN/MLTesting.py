import DataReading as DR
import torch 
import numpy as np
import matplotlib.pyplot as plt
from NNParameters import model, loss_fn



model.load_state_dict(torch.load("Trained_MLModel.pth"))
model.eval()

X,t = DR.readData("test")

X = torch.tensor(X, dtype=torch.float)
t = torch.tensor(t, dtype=torch.float).view(-1,1)

y_pred = model(X)

print(y_pred)

loss = loss_fn(y_pred,t)
print(loss)

y_pred = model(X).detach().numpy()

np.savetxt("../DataSets/OutPutProbability.txt", np.column_stack((t, y_pred)), delimiter=",")

