import torch, torch.nn as nn, torch.nn.functional as F
from .utils import init_weights, get_padding
from hw_code.losses import hifi_gan_loss as total_loss  # for the united model

# code inspired by the official HiFiGAN authors code: https://github.com/jik876/hifi-gan, although I have implemented many parts differently
# too many numbers to place them in config, so they are here


class ResBlockV1(nn.Module):

  def __init__(self, n_channels, kernel_size, dilations, lrelu_slope):  #kernel_size=3, dilations=[1, 3, 5], lrelu_slope=0.1
    super().__init__()
    self.stack1 = nn.Sequential(
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, dilations[0]), dilation=dilations[0])),
      nn.LeakyReLU(negative_slope=lrelu_slope),
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, 1), dilation=1)),
      nn.LeakyReLU(negative_slope=lrelu_slope),
    )
    self.stack1.apply(init_weights)
    self.stack2 = nn.Sequential(
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, dilations[1]), dilation=dilations[1])),
      nn.LeakyReLU(negative_slope=lrelu_slope),
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, 1), dilation=1)),
      nn.LeakyReLU(negative_slope=lrelu_slope),
    )
    self.stack2.apply(init_weights)
    self.stack3 = nn.Sequential(
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, dilations[2]), dilation=dilations[2])),
      nn.LeakyReLU(negative_slope=lrelu_slope),
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, 1), dilation=1)),
      nn.LeakyReLU(negative_slope=lrelu_slope),
    )
    self.stack3.apply(init_weights)

  def forward(self, x):
    x = x + self.stack1(x)
    x = x + self.stack2(x)
    x = x + self.stack3(x)
    return x


class ResBlockV2(nn.Module):

  def __init__(self, n_channels, kernel_size, dilations, lrelu_slope):  #kernel_size=3, dilations=[1, 3], lrelu_slope=0.1
    super().__init__()
    self.stack1 = nn.Sequential(
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, dilations[0]), dilation=dilations[0])),
      nn.LeakyReLU(negative_slope=lrelu_slope),
    )
    self.stack1.apply(init_weights)
    self.stack2 = nn.Sequential(
      nn.utils.weight_norm(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=get_padding(kernel_size, dilations[1]), dilation=dilations[1])),
      nn.LeakyReLU(negative_slope=lrelu_slope),
    )
    self.stack2.apply(init_weights)

  def forward(self, x):
    x = x + self.stack1(x)
    x = x + self.stack2(x)
    return x


class MultiReceptiveFieldFusion(nn.Module):

  def __init__(self, ResBlockType, n_channels, kernel_sizes, dilation_sizes, lrelu_slope):
    super().__init__()
    self.resblocks = nn.ModuleList([
      ResBlockType(n_channels=n_channels, kernel_size=kernel_size, dilations=dilations, lrelu_slope=lrelu_slope)
      for kernel_size, dilations in zip(kernel_sizes, dilation_sizes)
    ])

  def forward(self, x):
    x = [resblock(x) for resblock in self.resblocks]
    return sum(x) / len(x)


class HiFiGenerator(nn.Module):

  def __init__(self, in_channels, ResBlockType, lrelu_slope):
    super().__init__()
    in_channels, upsample_init_channels = 80, 256
    upsample_rates, upsample_kernel_sizes = [8, 8, 4], [16, 16, 8]
    mrff_kernel_sizes, mrff_dilation_sizes = [3, 5, 7], [[1, 2], [2, 6], [3, 12]]
    self.conv_pre = nn.Sequential(
      nn.utils.weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=upsample_init_channels, kernel_size=7, stride=1, padding=3)),
      nn.LeakyReLU(lrelu_slope),
    )
    self.num_upsamples = len(upsample_kernel_sizes)
    self.upsamples = nn.ModuleList([
      nn.ConvTranspose1d(in_channels=upsample_init_channels//2**i, out_channels=upsample_init_channels//2**(i+1),
                         kernel_size=kernel_size, stride=stride, padding=(kernel_size-stride)//2)
      for i, (kernel_size, stride) in enumerate(zip(upsample_kernel_sizes, upsample_rates))
    ])
    self.upsamples.apply(init_weights)
    self.mrffs = nn.ModuleList([
      MultiReceptiveFieldFusion(ResBlockType=ResBlockType, n_channels=upsample_init_channels//2**(i+1),
                                kernel_sizes=mrff_kernel_sizes, dilation_sizes=mrff_dilation_sizes, lrelu_slope=lrelu_slope)
      for i in range(self.num_upsamples)
    ])
    self.conv_post = nn.Sequential(
      nn.utils.weight_norm(nn.Conv1d(in_channels=upsample_init_channels//2**len(self.upsamples), out_channels=1, kernel_size=7, stride=1, padding=3)),
      nn.Tanh(),
    )
    self.conv_post.apply(init_weights)

  @torch.compile()
  def forward(self, x):
    x = self.conv_pre(x)
    for upsample, MRFF in zip(self.upsamples, self.mrffs):
      x = upsample(x)
      x = MRFF(x)
    x = self.conv_post(x)
    return x


class HiFiDiscriminatorP(nn.Module):

  def __init__(self, period, norm, lrelu_slope):
    super().__init__()
    self.period, self.lrelu_slope = period, lrelu_slope
    kernel_size, stride = 5, 3
    self.convs_main = nn.ModuleList([
      norm(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
      norm(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
      norm(nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
      norm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1), 0))),
      norm(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(2, 0)))
    ])
    self.conv_post = norm(nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))

  def forward(self, x):
    assert x.ndim == 3, f'HiFiDiscriminatorP got {x.shape=}, expected 3D'
    B, C, T = x.shape
    assert C == 1
    features = []
    n_pad = self.period - T % self.period
    x = F.pad(x, (0, n_pad), 'reflect')
    T += n_pad
    x = x.view(B, C, T // self.period, self.period)
    for conv in self.convs_main:
      x = conv(x)
      x = F.leaky_relu(x, self.lrelu_slope)
      features.append(x)
    x = self.conv_post(x).mean(dim=(1, 2, 3))
    assert x.ndim == 1  # batch
    features.append(x)
    return x, features


class HiFiDiscriminatorMultiPeriod(nn.Module):

  def __init__(self, lrelu_slope):
    super().__init__()
    self.children_discriminators = nn.ModuleList([
      HiFiDiscriminatorP(period=2, norm=nn.utils.weight_norm, lrelu_slope=lrelu_slope),
      HiFiDiscriminatorP(period=3, norm=nn.utils.weight_norm, lrelu_slope=lrelu_slope),
      HiFiDiscriminatorP(period=5, norm=nn.utils.weight_norm, lrelu_slope=lrelu_slope),
      HiFiDiscriminatorP(period=7, norm=nn.utils.weight_norm, lrelu_slope=lrelu_slope),
      HiFiDiscriminatorP(period=11, norm=nn.utils.weight_norm, lrelu_slope=lrelu_slope),
    ])

  @torch.compile()
  def forward(self, x):
    assert x.ndim == 3, f'HiFiDiscriminatorMultiPeriod got {x.shape=}, expected 3D'
    xs, features = [], []
    for child_discriminator in self.children_discriminators:
      x1, feats = child_discriminator(x)
      xs.append(x1)
      features.append(feats)
    return xs, features


class HiFiDiscriminatorS(nn.Module):

  def __init__(self, norm, lrelu_slope):
    super().__init__()
    self.lrelu_slope = lrelu_slope
    self.convs_main = nn.ModuleList([
      norm(nn.Conv1d(in_channels=1, out_channels=128, kernel_size=15, stride=1, padding=7)),
      norm(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=41, stride=2, padding=20, groups=4)),
      norm(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=41, stride=2, padding=20, groups=16)),
      norm(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=41, stride=4, padding=20, groups=16)),
      norm(nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=41, stride=4, padding=20, groups=16)),
      norm(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=41, stride=1, padding=20, groups=16)),
      norm(nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2))
    ])
    self.conv_post = norm(nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    assert x.ndim == 3, f'HiFiDiscriminatorS got {x.shape=}, expected 3D'
    features = []
    for conv in self.convs_main:
      x = conv(x)
      x = F.leaky_relu(x, self.lrelu_slope)
      features.append(x)
    x = self.conv_post(x).mean(dim=(1, 2))
    assert x.ndim == 1  # batch
    features.append(x)
    return x, features


class HiFiDiscriminatorMultiScale(nn.Module):

  def __init__(self, lrelu_slope):
    super().__init__()
    self.children_discriminators = nn.ModuleList([
      HiFiDiscriminatorS(norm=nn.utils.spectral_norm, lrelu_slope=lrelu_slope),
      HiFiDiscriminatorS(norm=nn.utils.weight_norm, lrelu_slope=lrelu_slope),
      HiFiDiscriminatorS(norm=nn.utils.weight_norm, lrelu_slope=lrelu_slope),
    ])
    #self.avg_pools = nn.ModuleList([
    #  nn.Identity(),
    #  nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
    #  nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
    #])

  @torch.compile()
  def forward(self, x):
    assert x.ndim == 3, f'HiFiDiscriminatorMultiScale got {x.shape=}, expected 3D'
    xs, features = [], []
    #for child_discriminator, avg_pool in zip(self.children_discriminators, self.avg_pools):
    for child_discriminator in self.children_discriminators:
      x1, feats = child_discriminator(x)
      #x1 = avg_pool(x1)
      xs.append(x1)
      features.append(feats)
    return xs, features


class HiFiGAN(nn.Module):  # this guy is my own, it encompasses all the required three components

  def __init__(self, audio_to_mel_fn, n_mels=80, lrelu_slope=0.1):
    super().__init__()
    self.audio_to_mel_fn = audio_to_mel_fn
    self.generator = HiFiGenerator(in_channels=n_mels, ResBlockType=ResBlockV2, lrelu_slope=lrelu_slope)
    self.disc1 = HiFiDiscriminatorMultiPeriod(lrelu_slope=lrelu_slope)
    self.disc2 = HiFiDiscriminatorMultiScale(lrelu_slope=lrelu_slope)

  def forward(self, real_audio, real_mel):  # RETURNS LOSS, it is for convenience, because the loss is complex and highly depends on model types and relations,
                                            # and we would have to refactor training pipeline to support that otherwise
    assert real_audio.ndim == 3, f'HiFiGAN got {real_audio.shape=}, expected 3D'
    assert real_mel.ndim == 3, f'HiFiGAN got {real_mel.shape=}, expected 3D'
    #print('running generator')
    fake_audio = self.generator(real_mel)
    assert fake_audio.shape == real_audio.shape, f'HiFiGenerator produced audio of shape {fake_audio.shape} while real audio has shape {real_audio.shape}'
    assert fake_audio.shape[1] == 1
    #print('audio to mel')
    fake_mel = self.audio_to_mel_fn(fake_audio[:, 0])
    assert fake_mel.shape == real_mel.shape, f'fake_audio -> fake_mel produced mel of shape {fake_mel.shape} while real mel has shape {real_mel.shape}'
    #print('running discriminators')
    disc1_output_on_real_audio__D, disc1_features_on_real_audio__D = self.disc1(real_audio)  # grad only wrt D1
    disc2_output_on_real_audio__D, disc2_features_on_real_audio__D = self.disc2(real_audio)  # grad only wrt D2
    disc1_output_on_fake_audio__D, disc1_features_on_fake_audio__D = self.disc1(fake_audio)  # grad only wrt D1
    disc2_output_on_fake_audio__D, disc2_features_on_fake_audio__D = self.disc2(fake_audio)  # grad only wrt D2
    with torch.inference_mode():
      disc1_output_on_real_audio__G, disc1_features_on_real_audio__G = self.disc1(real_audio)  # grad only wrt G
      disc2_output_on_real_audio__G, disc2_features_on_real_audio__G = self.disc2(real_audio)  # grad only wrt G
      disc1_output_on_fake_audio__G, disc1_features_on_fake_audio__G = self.disc1(fake_audio)  # grad only wrt G
      disc2_output_on_fake_audio__G, disc2_features_on_fake_audio__G = self.disc2(fake_audio)  # grad only wrt G
    #print('computing loss')
    return total_loss(real_audio, fake_audio, real_mel, fake_mel,
                      disc1_output_on_real_audio__G, disc1_features_on_real_audio__G, disc2_output_on_real_audio__G, disc2_features_on_real_audio__G,
                      disc1_output_on_real_audio__D, disc1_features_on_real_audio__D, disc2_output_on_real_audio__D, disc2_features_on_real_audio__D,
                      disc1_output_on_fake_audio__G, disc1_features_on_fake_audio__G, disc2_output_on_fake_audio__G, disc2_features_on_fake_audio__G,
                      disc1_output_on_fake_audio__D, disc1_features_on_fake_audio__D, disc2_output_on_fake_audio__D, disc2_features_on_fake_audio__D)

  @torch.inference_mode()
  def inference(self, mel):
    audio = self.generator(mel)
    return audio
