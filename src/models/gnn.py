class GNN(nn.Module):
    ''' A simple fully connected neural network with one hidden layer '''

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int) -> None:
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x