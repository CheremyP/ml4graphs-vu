class GraphAttentionLayer(nn.Module):
    """ A graph attention layer (GAT) """
    def __init__(self, in_features:int, out_features:int, dropout:float=0.6, alpha:float=0.2, concat:bool=True) -> None:
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        # Learnable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h:torch.Tensor, adj:torch.Tensor) -> torch.Tensor:
        Wh = torch.mm(h, self.W)  # Linear transformation
        N = h.size()[0]

        # Self-attention mechanism
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=self.alpha)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Aggregation
        h_prime = torch.matmul(attention, Wh)
        h_prime = F.elu(h_prime)  # Activation function

        return h_prime


class GAT(nn.Module):
    """ A graph attention network (GAT) with multiple attention heads. """
    def __init__(self, in_features:int, out_features:int, num_heads:int, dropout:float=0.6, alpha:float=0.2) -> None:
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # List of attention layers
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha) for _ in range(num_heads)
        ])

    def forward(self, data: object) -> torch.Tensor:
        h, edges_matrix = data.x, data.edge_index
        # Construct the adjacency matrix
        adjacency_matrix = torch.zeros(data.num_nodes, data.num_nodes)
        for edge in zip(edges_matrix[0], edges_matrix[1]):
            src, tgt = edge
            adjacency_matrix[src, tgt] = 1

        adjacency_matrix = adjacency_matrix.long()

        # Move tensors to GPU device
        h = h.to(device)
        adjacency_matrix = adjacency_matrix.to(device)

        # Stacking multiple attention heads
        all_head_outputs = [layer(h, adjacency_matrix) for layer in self.attention_layers]
        output = torch.mean(torch.stack(all_head_outputs), dim=0)

        return output