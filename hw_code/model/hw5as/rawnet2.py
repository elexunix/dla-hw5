import torch, torch.nn as nn
from .sinc import SincConvEpic

# this is a minimalistic implementation of https://arxiv.org/pdf/2011.01108.pdf


class FMS(nn.Module):  # based on https://arxiv.org/pdf/2004.00526.pdf, which is the 7-th source of the RawNet2 paper

  def __init__(self, n_channels):
    super().__init__()
    # "We derive a scale vector to
    #  conduct FMS by first performing global average pooling on the
    #  time axis, and then feed-forwarding through a fully-connected
    #  layer followed by sigmoid activation."
    self.scaler = nn.Sequential(
      nn.Linear(in_features=n_channels, out_features=n_channels),
      nn.Sigmoid(),
    )

  def forward(self, x):
    assert x.ndim == 3           # x.shape == (B, C, L)
    s = self.scaler(x.mean(-1))  # s.shape == (B, C, 1)
    return (x + 1) * s.unsqueeze(-1)


class RawNetBlock(nn.Module):

  def __init__(self, in_channels, out_channels, lrelu_slope):
    super().__init__()
    self.pre_stack = nn.Sequential(
      nn.BatchNorm1d(num_features=in_channels),
      nn.LeakyReLU(negative_slope=lrelu_slope),
    )
    self.stack = nn.Sequential(
      nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
      nn.BatchNorm1d(num_features=out_channels),
      nn.LeakyReLU(negative_slope=lrelu_slope),
      nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
    )
    self.skip_connection = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    self.post_stack = nn.Sequential(
      nn.MaxPool1d(kernel_size=3),
      FMS(n_channels=out_channels),
    )

  def forward(self, x):
    y = self.pre_stack(x)
    x = self.stack(y) + self.skip_connection(y)
    x = self.post_stack(x)
    return x


class RawNet2Model(nn.Module):

  def __init__(self, n_channels=[128, 20, 128, 128, 128, 128], lrelu_slope=0.3, sample_rate=16000, sinc_spacing_type='mel'):
    super().__init__()
    self.stack = nn.Sequential(
      #nn.Conv1d(in_channels=1, out_channels=n_channels[0], kernel_size=1024),
      SincConvEpic(in_channels=1, out_channels=n_channels[0], kernel_size=1023, sample_rate=sample_rate, spacing_type=sinc_spacing_type),
      nn.MaxPool1d(kernel_size=3),
      nn.BatchNorm1d(num_features=n_channels[0]),
      nn.LeakyReLU(negative_slope=lrelu_slope),
      RawNetBlock(in_channels=n_channels[0], out_channels=n_channels[1], lrelu_slope=lrelu_slope),
      RawNetBlock(in_channels=n_channels[1], out_channels=n_channels[2], lrelu_slope=lrelu_slope),
      RawNetBlock(in_channels=n_channels[2], out_channels=n_channels[3], lrelu_slope=lrelu_slope),
      RawNetBlock(in_channels=n_channels[3], out_channels=n_channels[4], lrelu_slope=lrelu_slope),
      RawNetBlock(in_channels=n_channels[4], out_channels=n_channels[5], lrelu_slope=lrelu_slope),
      nn.BatchNorm1d(num_features=n_channels[5]),
      nn.LeakyReLU(negative_slope=lrelu_slope),
    )
    self.gru = nn.GRU(input_size=n_channels[5], hidden_size=1024, num_layers=3, batch_first=True)
    self.fc = nn.Linear(in_features=1024, out_features=2)

  def forward(self, x):
    assert x.ndim == 3        # x.shape == (B, 1, L)
    x = self.stack(x)         # x.shape == (B, 128, L')
    x = x.transpose(-1, -2)   # x.shape == (B, L', 128)
    x = self.gru(x)[1][-1]    # x.shape == (B, 1024)
    x = self.fc(x)            # x.shape == (B, 2)
    return x

