import torch, torch.nn as nn


class Encoder(nn.Module):

  def __init__(self, 


class FastSpeech2Model(nn.Module):

  def __init__(self, 
    self.encoder = Encoder()
    self.variance_adaptor = VarianceAdaptor()
    self.decoder = Decoder()

  def forward(self, x, lens):
    masks = get_masks(lens)
    x = self.encoder(x, masks)
    x = self.variance_adaptor(x)
    x = self.decoder(x)
    x = self.mel_head(x)
    return x, x + waveform_head(x)
