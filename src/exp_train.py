import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from models import GraphAttentionV2Layer
import torch.nn as nn
# Set the path where the dataset will be stored
path = 'data'



# Training parameters
nb_epochs = 10000
lr = 0.005  # learning rate

n_heads = [8, 1]  # additional entry for the output layer

# Define the input, hidden, and output dimensions
input_dim = dataset.num_features
hidden_dim = 64

# Define the model
model = GraphAttentionV2Layer(in_features=input_dim, out_features=hidden_dim, n_heads=n_heads) # Select the model to use

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the appropriate device

def train(dataset):
    # Load the Cora dataset
    dataset = Planetoid(root=path, name='Cora', split='random')

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(nb_epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)  # Move data to the appropriate device
            optimizer.zero_grad()
            x, y = data.x, data.y
            output = model(data)
            loss = criterion(output[data.train_mask], y[data.train_mask])  # Use only the training mask
            loss.backward()
            optimizer.step()

        # Print the loss after each epoch
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    return dataset, model, device

def eval():
    # Evaluate on the test set
    model.eval()

    data = dataset[0].to(device)  # Move data to the appropriate device for evaluation

    # Pass the features through the model for prediction
    pred = model(data).argmax(dim=1)

    # Calculate accuracy
    correct = (pred[data.test_mask] == y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

if __name__ == '__main__':
    acc = []
    for i in range(20):
        dataset, model , _ = train()
        acc = eval(dataset)
        print(f'Accuracy: {acc:.4f}')
        acc.append(acc)

    print(f'Average accuracy: {sum(acc) / len(acc):.4f}')


