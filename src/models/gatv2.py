class GATv2(nn.Module):
    """ A graph attention network version 2 (GATv2) with multiple attention heads. """
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False,
                 flash:bool = True) -> None:
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.ModuleList([nn.Linear(in_features, self.n_hidden, bias=False) for _ in range(n_heads)])
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.ModuleList([nn.Linear(in_features, self.n_hidden, bias=False) for _ in range(n_heads)])
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        self.flash = flash
        if flash and not torch.backends.cuda.flash_sdp_enabled():
            torch.backends.cuda.enable_flash_sdp(True)

    def forward(self, data:object) -> torch.Tensor:
        h, edges_matrix = data.x, data.edge_index
        n_nodes = h.shape[0]

        # Construct the adjacency matrix
        adjacency_matrix = torch.zeros(data.num_nodes, data.num_nodes)
        for edge in zip(edges_matrix[0], edges_matrix[1]):
            src, tgt = edge
            adjacency_matrix[src, tgt] = 1

        adjacency_matrix = adjacency_matrix.long()

        # Reshape adjacency matrix to match the shape of e
        adjacency_matrix = adjacency_matrix.unsqueeze(2)

        # Move tensors to GPU device
        h = h.to(device)
        adjacency_matrix = adjacency_matrix.to(device)

        g_l = torch.stack([linear(h) for linear in self.linear_l], dim=1)
        g_r = torch.stack([linear(h) for linear in self.linear_r], dim=1)

        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat(1, 1, n_nodes, 1)

        # Reshape g_l_repeat to match g_r_repeat_interleave
        g_l_repeat = g_l_repeat.view(*g_r_repeat_interleave.shape)

        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        assert adjacency_matrix.shape[0] == 1 or adjacency_matrix.shape[0] == n_nodes
        assert adjacency_matrix.shape[1] == 1 or adjacency_matrix.shape[1] == n_nodes
        assert adjacency_matrix.shape[2] == 1 or adjacency_matrix.shape[2] == self.n_heads

        # Mask the attention scores
        e = e.masked_fill(adjacency_matrix == 0, float('-inf'))

        a = self.softmax(e)
        a = self.dropout(a)

        # Transpose the dimensions of 'a' to match with 'g_r' for matrix multiplication

        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)
