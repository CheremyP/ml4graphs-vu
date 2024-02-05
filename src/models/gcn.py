class GCN(torch.nn.Module):
    """" A graph convolutional network (GCN) model 
    """

    def __init__(self, num_node_features:int, num_classes:int) -> None:
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index ) -> torch.Tensor:

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x
