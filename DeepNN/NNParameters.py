import torch


n_hidden_neurons = 200
n_epochs = 1000
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-7

#10 hidden layers
model = torch.nn.Sequential(torch.nn.Linear(2, n_hidden_neurons), 
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(), 
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,n_hidden_neurons),
                            torch.nn.ReLU(),
                            torch.nn.Linear(n_hidden_neurons,1))

