import torch, torch.nn as nn


class ChannelwiseLayernorm(nn.LayerNorm):

  def __init__(self, n_channels):
    super().__init__(normalized_shape=n_channels)

  def forward(self, x):
    assert x.ndim == 3
    # x: (B, C, T)
    # according to the docs, nn.LayerNorm acts on the last dim, in our case we transpose to get the C dim there
    return super().forward(x.transpose(1, 2)).transpose(1, 2)  # of course, we transpose back later


#class GlobalLayernorm(nn.LayerNorm):
#
#  def __init__(self, n_channels, n_moments)
#    super().__init__(normalized_shape=n_channels*n_moments)
#
#  def forward(self, x):
#    assert x.ndim == 3
#    x = x.reshape(x.shape[0], -1)
#    x = super().forward(x)
#    return x.reshape(x.shape[0], n_channels, n_moments)

# ^ it CANNOT be done this way, as we don't know n_moments beforehand
# So it has to be done this way:

class GlobalLayernorm(nn.Module):

  def __init__(self, n_channels, eps=1e-5):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(n_channels, 1))
    self.beta = nn.Parameter(torch.zeros(n_channels, 1))
    self.eps = eps

  def forward(self, x):
    assert x.ndim == 3
    # x: (B, C, T)
    var, mean = torch.var_mean(x, (1, 2), correction=0, keepdim=True)
    return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
