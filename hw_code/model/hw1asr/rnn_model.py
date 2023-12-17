import torch
from torch import nn
import torch.nn.functional as F

from hw_code.base import BaseModel


class RNNModel(BaseModel):
  def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
    super().__init__(n_feats, n_class, **batch)
    self.rnn = nn.RNN(input_size=n_feats, hidden_size=fc_hidden, num_layers=3, batch_first=True)
    self.head = nn.Linear(in_features=fc_hidden, out_features=n_class)

  def forward(self, spectrogram, **batch):
    output, h_n = self.rnn(spectrogram.transpose(1, 2))
    return {"logits": self.head(output)}

  def transform_input_lengths(self, input_lengths):
    return input_lengths # we don't reduce time dimension here


class LSTMModel(BaseModel):
  def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
    super().__init__(n_feats, n_class, **batch)
    n_channels = n_feats
    self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=5, padding=2)
    self.conv2 = nn.Conv1d(in_channels=n_channels, out_channels=2*n_channels, kernel_size=5, padding=2)
    self.conv3 = nn.Conv1d(in_channels=2*n_channels, out_channels=4*n_channels, kernel_size=5, padding=2)
    self.lstm = nn.LSTM(input_size=8*n_channels, hidden_size=fc_hidden, bidirectional=True, num_layers=3, batch_first=True)
    self.head = nn.Linear(in_features=2*fc_hidden, out_features=n_class)

  def forward(self, spectrogram, **batch):
    x1 = F.relu(self.conv1(spectrogram))
    x2 = F.relu(self.conv2(x1))
    x3 = F.relu(self.conv3(x2))
    x = torch.cat([spectrogram, x1, x2, x3], dim=1)
    output, h_n = self.lstm(x.transpose(1, 2))
    #output = F.pad(output, (0, 0, 0, spectrogram.shape[2] - output.shape[1]))
    return {"logits": self.head(output)}

  def transform_input_lengths(self, input_lengths):
    return input_lengths # we don't reduce time dimension here
