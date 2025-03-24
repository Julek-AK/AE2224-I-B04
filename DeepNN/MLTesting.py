import DataReading as DR
import torch 
import numpy as np
import matplotlib.pyplot as plt
from NNParameters import model, loss_fn
from sklearn.preprocessing import StandardScaler

model.load_state_dict(torch.load("Trained_MLModel.pth"))
model.eval()

X,t = DR.readData2("test")

#Normalise data 
scaler = StandardScaler()
X = scaler.fit_transform(X)
t = scaler.fit_transform(np.array(t).reshape(-1,1)).flatten()

X = torch.tensor(X, dtype=torch.float)
t = torch.tensor(t, dtype=torch.float).view(-1,1)

y_pred = model(X)

print(y_pred)

loss = loss_fn(y_pred,t)
print(loss)

y_pred = model(X).detach().numpy()

#Denormalise
t = DR.readData2("test")[1]
scaler = StandardScaler()
scaler = scaler.fit(np.array(t).reshape(-1,1))
y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
y_pred = y_pred.reshape(-1)



np.savetxt("../DataSets/OutPutProbability.txt", np.column_stack((t, y_pred)), delimiter=",")

