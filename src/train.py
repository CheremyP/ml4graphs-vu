import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from models import GraphAttentionV2Layer
import torch.nn as nn
# Set the path where the dataset will be stored
path = 'data'

# Load the Cora dataset
dataset = Planetoid(root=path, name='Cora')
dataset_2 = Planetoid(root=path, name='citeseer')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training parameters
batch_size = 4
nb_epochs = 10000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8]  # numbers of hidden units per each attention head in each layer
n_heads = [8, 1]  # additional entry for the output layer
residual = False

DROPOUT_RATE: float = 0.5

# Define the input, hidden, and output dimensions
input_dim = dataset.num_features
hidden_dim = 64
output_dim = dataset.num_classes

# Define the model
model = GraphAttentionV2Layer(in_features=input_dim, out_features=hidden_dim, n_heads=n_heads)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train():
    for epoch in range(nb_epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        # Print loss for every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def eval():
    # Evaluate on the test set
    model.eval()

    data = dataset[0].to(device)  # Move data to the appropriate device for evaluation

    # Pass the features through the model for prediction
    pred = model(data).argmax(dim=1)

    # Calculate accuracy
    correct = (pred[data.test_mask] == y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')


if __name__ == '__main__':
    train()
    eval()

