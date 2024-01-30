import torch
from torch import nn
from torch.utils.data import DataLoader
from plantoid.datasets import CoraDataset

class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
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

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        n_nodes = h.shape[0]
        g_l = torch.stack([linear(h) for linear in self.linear_l], dim=1)
        g_r = torch.stack([linear(h) for linear in self.linear_r], dim=1)

        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat(1, 1, n_nodes, 1)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.matmul(a.transpose(1, 2), g_r)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)


# Cora data loader
dataset = CoraDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model = GraphAttentionV2Layer(in_features=dataset.num_features, out_features=64, n_heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        h, adj_mat, labels = batch
        output = model(h, adj_mat)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
