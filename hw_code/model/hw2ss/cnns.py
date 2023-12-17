import torch, torch.nn as nn
from .layernorms import GlobalLayernorm


class TCNBlockWithSpeaker(nn.Module):

  def __init__(self, in_channels, speaker_embed_dim, inner_channels, kernel_size):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels=in_channels+speaker_embed_dim, out_channels=inner_channels, kernel_size=1)
    self.prelu1 = nn.PReLU()
    self.norm1 = GlobalLayernorm(n_channels=inner_channels)
    assert kernel_size % 2 == 1
    self.conv2 = nn.Conv1d(in_channels=inner_channels, out_channels=inner_channels, kernel_size=kernel_size, groups=inner_channels, padding=(kernel_size-1)//2)
    self.prelu2 = nn.PReLU()
    self.norm2 = GlobalLayernorm(n_channels=inner_channels)
    self.conv3 = nn.Conv1d(in_channels=inner_channels, out_channels=in_channels, kernel_size=1)

  def forward(self, x, aux):
    # x: (B, C, T)
    # aux: (B, D)
    # output: (B, C, T)
    assert x.ndim == 3
    B, C, T = x.shape
    y = torch.cat([x, aux.unsqueeze(-1).repeat(1, 1, T)], 1)  # (B, C + D, T)
    y = self.conv1(y)
    y = self.prelu1(y)
    y = self.norm1(y)
    y = self.conv2(y)
    y = self.prelu2(y)
    y = self.norm2(y)
    y = self.conv3(y)
    return x + y


class TCNBlockWithoutSpeaker(TCNBlockWithSpeaker):

  def __init__(self, in_channels, inner_channels, kernel_size):
    super().__init__(in_channels=in_channels, speaker_embed_dim=0, inner_channels=inner_channels, kernel_size=kernel_size)

  def forward(self, x):
    B, C, T = x.shape
    return TCNBlockWithSpeaker.forward(self, x, x[:0].reshape(B, 0))


class ResNetBlock(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm1d(num_features=out_channels)
    self.prelu1 = nn.PReLU()
    self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)
    self.bn2 = nn.BatchNorm1d(num_features=out_channels)
    self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
    self.prelu2 = nn.PReLU()
    self.pool = nn.MaxPool1d(kernel_size=3, padding=1)

  def forward(self, x):
    # x: (B, Cin, T)
    # output: (B, Cout, T)
    assert x.ndim == 3
    y = self.conv1(x)
    y = self.bn1(y)
    y = self.prelu1(y)
    y = self.conv2(y)
    y = self.bn2(y)
    y = y + self.conv3(x)
    y = self.prelu2(y)
    return self.pool(y)
