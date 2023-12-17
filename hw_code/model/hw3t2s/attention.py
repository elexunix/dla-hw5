import torch, torch.nn as nn


class AttentionLayer(nn.Module):

  def forward(self, Q, K, V, mask=None):
    assert Q.ndim, K.ndim, V.ndim == 3, 3, 3
    D = Q.shape[1]
    attention = Q @ K.transpose(-1, -2) / np.sqrt(D)
    attention = attention.masked_fill(mask, -np.inf) if mask is not None else attention
    return attention.softmax(-1) @ V, attention


class MultiHeadAttentionLayer(nn.Module):  # implementation based on my SHW4 code in our DL2 course

  def __init__(self, n_heads, embed_dim):
    super().__init__()
    assert embed_dim % n_heads == 0, "Embedding dimension must be 0 modulo number of heads"
    self.n_heads = n_heads
    self.embed_dim = embed_dim
    self.head_dim = embed_dim // n_heads

    self.q_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.k_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.v_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    self.o_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    nn.init.xavier_uniform_(self.q_proj.weight)
    nn.init.xavier_uniform_(self.k_proj.weight)
    nn.init.xavier_uniform_(self.v_proj.weight)
    nn.init.xavier_uniform_(self.o_proj.weight)

  def forward(self, x, mask=None):
    B, L, D = x.shape

    Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    attention = Q @ K.transpose(-1, -2) / np.sqrt(D)
    attention = attention.masked_fill(mask, -np.inf) if mask is not None else attention
    return self.o_proj(attention.transpose(-1, -2).reshape(B, L, self.embed_dim)), attention
