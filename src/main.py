import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

#TODO: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html

# Set the path where the dataset will be stored
path = '/home/cheremy/Documents/personal/ml4graphs/project/data'

# Load the Cora dataset
dataset = Planetoid(root=path, name='Cora')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False

class SimpleModel(nn.Module):
    ''' A simple fully connected neural network with one hidden layer '''


    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  

class GCN(torch.nn.Module):
    ''' A graph convolutional network (GCN) model '''
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=DROPOUT_RATE)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, # For output we only use one head
                             concat=False, dropout=DROPOUT_RATE)


    def forward(self, x, edge_index):
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class GAT(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(GAT, self).__init__()
        self.g = g
        self.num_heads = num_heads
        self.merge = merge

        # Initialize GAT heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(self.build_head(in_dim, out_dim))

    def build_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.Linear(2 * out_dim, 1, bias=False)
        )

    def reset_parameters(self):
        for head in self.heads:
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(head[0].weight, gain=gain)
            nn.init.xavier_normal_(head[1].weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = F.leaky_relu(self.heads[0][1](z2))  # Assuming using the first head for edge attention
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        head_outs = [self.process_head(attn_head, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))

    def process_head(self, head, h):
        z = head[0](h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


DROPOUT_RATE: float = 0.5
# Define the input, hidden, and output dimensions
input_dim = dataset.num_features
hidden_dim = 64
output_dim = dataset.num_classes

# Create an instance of the SimpleModel
model = SimpleModel(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

# Training loop
for epoch in range(nb_epochs):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        x, y = data.x, data.y
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Print the loss after each epoch
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# After training, you can use the model for predictions
model.eval()
with torch.no_grad():
    for data in train_loader:
        x, y = data.x, data.y
        output = model(x)
        predicted_labels = torch.argmax(output, dim=1)
        print(f"Predicted Labels: {predicted_labels}")

# Create an instance of the GAT model
model = GAT(input_dim, hidden_dim, output_dim, n_heads)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

# Training loop
for epoch in range(nb_epochs):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        x, edge_index, y = data.x, data.edge_index, data.y
        output = model(x, edge_index)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Print the loss after each epoch
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# After training, you can use the model for predictions
model.eval()
with torch.no_grad():
    for data in train_loader:
        x, edge_index, y = data.x, data.edge_index, data.y
        output = model(x, edge_index)
        predicted_labels = torch.argmax(output, dim=1)
        print(f"Predicted Labels: {predicted_labels}")