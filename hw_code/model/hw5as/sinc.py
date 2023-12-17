import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class SincConvEpic(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, sample_rate, spacing_type='mel'):
    super().__init__()
    assert in_channels == 1, "Audio should have exactly 1 channel"
    assert kernel_size % 2 == 1, "SincConv kernel size should be odd"
    self.out_channels, self.kernel_size, self.sample_rate = out_channels, kernel_size, sample_rate
    self.cnt = kernel_size // 2

    self.lowest_f1, self.lowest_f2 = 0, 0  #50, 100
    min_f, max_f = 0, sample_rate / 2 - self.lowest_f2  # 30
    if spacing_type == 'mel':
      initial_freqs = self.mel_to_hz(torch.linspace(self.hz_to_mel(min_f), self.hz_to_mel(max_f), out_channels + 1))  # equally spaced in Mels
    elif spacing_type == 'hz':
      initial_freqs = torch.linspace(min_f, max_f, out_channels + 1)                                                  # equally spaced in Hz

    self.trainable_f1s = nn.Parameter(initial_freqs[:-1, None])  # shape (out_channels, 1)
    self.trainable_f2s = nn.Parameter(initial_freqs[1:, None])   # shape (out_channels, 1)

  @staticmethod
  def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

  @staticmethod
  def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

  def forward(self, waveforms):
    f1s = self.lowest_f1 + torch.abs(self.trainable_f1s)                                                                       # shape (out_channels, 1)
    f2s = self.lowest_f2 + torch.abs(self.trainable_f2s) - torch.abs(self.trainable_f1s)                                       # shape (out_channels, 1)

    self.hamming_window = torch.hamming_window(self.kernel_size, device=waveforms.device)
    tt = torch.arange(-self.cnt, self.cnt + 1, device=waveforms.device)[None] / self.sample_rate                               # shape (1, K)
    band_pass = (torch.sinc(2*np.pi * f2s * tt) * f2s - torch.sinc(2*np.pi * f1s * tt) * f1s) * self.hamming_window            # shape (out_channels, K)

    return F.conv1d(waveforms, band_pass[:, None])
