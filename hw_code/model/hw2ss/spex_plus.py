import torch, torch.nn as nn, torch.nn.functional as F

from .layernorms import ChannelwiseLayernorm, GlobalLayernorm
from .cnns import TCNBlockWithoutSpeaker, TCNBlockWithSpeaker, ResNetBlock


class SharedEncoder(nn.Module):

  def __init__(self, n_channels, ks_short, ks_middle, ks_long, stride):
    super().__init__()
    self.encoder_short = nn.Conv1d(in_channels=1, out_channels=n_channels, kernel_size=ks_short, stride=stride)
    self.encoder_middle = nn.Conv1d(in_channels=1, out_channels=n_channels, kernel_size=ks_middle, stride=stride)
    self.encoder_long = nn.Conv1d(in_channels=1, out_channels=n_channels, kernel_size=ks_long, stride=stride)

  def forward(self, x):
    # x: (B, T)
    assert x.ndim == 2
    x = x.unsqueeze(1)
    encoded_short = self.encoder_short(x)
    encoded_middle = self.encoder_middle(x)
    encoded_long = self.encoder_long(x)
    T1, T2, T3 = encoded_short.shape[-1], encoded_middle.shape[-1], encoded_long.shape[-1]
    assert T1 >= T2 >= T3
    encoded_middle = F.pad(encoded_middle, ((T1 - T2) // 2, T1 - T2 - (T1 - T2) // 2))
    encoded_long = F.pad(encoded_long, ((T1 - T3) // 2, T1 - T3 - (T1 - T3) // 2))
    x = F.relu(torch.cat([encoded_short, encoded_middle, encoded_long], 1))
    return x, encoded_short, encoded_middle, encoded_long


class SpExPlusModel(nn.Module):

  def __init__(self, n_possible_train_speakers, n_tcns_per_block=8, n_channels_shared=256, n_channels_stem=256, n_channels_inner=512, speaker_embed_dim=256, kernel_size=3,
               ks_short=20, ks_middle=80, ks_long=160, stride=10):
    super().__init__()
    self.shared_encoder = SharedEncoder(n_channels=n_channels_shared, ks_short=ks_short, ks_middle=ks_middle, ks_long=ks_long, stride=stride)
    # Speaker Extractor (the left network)
    self.speaker_extractor_prefix = nn.Sequential(
      ChannelwiseLayernorm(n_channels=3*n_channels_shared),
      nn.Conv1d(in_channels=3*n_channels_shared, out_channels=n_channels_stem, kernel_size=1),
    )
    self.conv_block_1a = TCNBlockWithSpeaker(in_channels=n_channels_stem, speaker_embed_dim=speaker_embed_dim, inner_channels=n_channels_inner, kernel_size=kernel_size)
    self.conv_block_1b = nn.Sequential(*[TCNBlockWithoutSpeaker(in_channels=n_channels_stem, inner_channels=n_channels_inner, kernel_size=kernel_size) for i in range(n_tcns_per_block)])
    self.conv_block_2a = TCNBlockWithSpeaker(in_channels=n_channels_stem, speaker_embed_dim=speaker_embed_dim, inner_channels=n_channels_inner, kernel_size=kernel_size)
    self.conv_block_2b = nn.Sequential(*[TCNBlockWithoutSpeaker(in_channels=n_channels_stem, inner_channels=n_channels_inner, kernel_size=kernel_size) for i in range(n_tcns_per_block)])
    self.conv_block_3a = TCNBlockWithSpeaker(in_channels=n_channels_stem, speaker_embed_dim=speaker_embed_dim, inner_channels=n_channels_inner, kernel_size=kernel_size)
    self.conv_block_3b = nn.Sequential(*[TCNBlockWithoutSpeaker(in_channels=n_channels_stem, inner_channels=n_channels_inner, kernel_size=kernel_size) for i in range(n_tcns_per_block)])
    self.conv_block_4a = TCNBlockWithSpeaker(in_channels=n_channels_stem, speaker_embed_dim=speaker_embed_dim, inner_channels=n_channels_inner, kernel_size=kernel_size)
    self.conv_block_4b = nn.Sequential(*[TCNBlockWithoutSpeaker(in_channels=n_channels_stem, inner_channels=n_channels_inner, kernel_size=kernel_size) for i in range(n_tcns_per_block)])
    self.get_mask_short = nn.Sequential(
      nn.Conv1d(in_channels=n_channels_stem, out_channels=n_channels_shared, kernel_size=1),
      nn.ReLU(),
    )
    self.decoder_short = nn.ConvTranspose1d(in_channels=n_channels_shared, out_channels=1, kernel_size=ks_short, stride=stride)
    self.get_mask_middle = nn.Sequential(
      nn.Conv1d(in_channels=n_channels_stem, out_channels=n_channels_shared, kernel_size=1),
      nn.ReLU(),
    )
    self.decoder_middle = nn.ConvTranspose1d(in_channels=n_channels_shared, out_channels=1, kernel_size=ks_middle, stride=stride)
    self.get_mask_long = nn.Sequential(
      nn.Conv1d(in_channels=n_channels_stem, out_channels=n_channels_shared, kernel_size=1),
      nn.ReLU(),
    )
    self.decoder_long = nn.ConvTranspose1d(in_channels=n_channels_shared, out_channels=1, kernel_size=ks_long, stride=stride)
    # Speaker Encoder (the right network)
    self.speaker_encoder_prefix = nn.Sequential(
      ChannelwiseLayernorm(n_channels=3*n_channels_shared),
      nn.Conv1d(in_channels=3*n_channels_shared, out_channels=n_channels_stem, kernel_size=1),
      ResNetBlock(in_channels=n_channels_stem, out_channels=n_channels_stem),
      ResNetBlock(in_channels=n_channels_stem, out_channels=n_channels_inner),
      ResNetBlock(in_channels=n_channels_inner, out_channels=n_channels_inner),
      nn.Conv1d(in_channels=n_channels_inner, out_channels=speaker_embed_dim, kernel_size=1),
    )
    assert kernel_size % 2 == 1
    self.speaker_encoder_head = nn.Linear(in_features=speaker_embed_dim, out_features=n_possible_train_speakers)

  def forward(self, x, ref):
    # x: (B, T0)
    # ref: (B, T1)
    y, y1, y2, y3 = self.shared_encoder(x)
    B, Cy, Ty = y.shape
    ref, _, _, _ = self.shared_encoder(ref)
    y = self.speaker_extractor_prefix(y)
    ref = self.speaker_encoder_prefix(ref)
    ref = ref.mean(-1)  # (B, D)
    y = self.conv_block_1b(self.conv_block_1a(y, ref))
    y = self.conv_block_2b(self.conv_block_2a(y, ref))
    y = self.conv_block_3b(self.conv_block_3a(y, ref))
    y = self.conv_block_4b(self.conv_block_4a(y, ref))
    decoded_short = self.decoder_short(self.get_mask_short(y) * y1)
    decoded_middle = self.decoder_middle(self.get_mask_middle(y) * y2)
    decoded_long = self.decoder_long(self.get_mask_long(y) * y3)
    T1, T2, T3 = decoded_short.shape[-1], decoded_middle.shape[-1], decoded_long.shape[-1]
    assert T1 <= T2 <= T3
    return decoded_short[:, 0], decoded_middle[:, 0, :T1], decoded_long[:, 0, :T1], self.speaker_encoder_head(ref)
