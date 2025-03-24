import torch


n_hidden_neurons = 126
n_epochs = 10000
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 0.01

#10 hidden layers
model = torch.nn.Sequential(torch.nn.Linear(4, n_hidden_neurons), 
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(), 
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,1))

