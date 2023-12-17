import os, pathlib, glob
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import librosa
from hw_code.utils import ROOT_PATH


@dataclass
class MelSpectrogramConfig:
  sr: int = 22050
  win_length: int = 1024
  hop_length: int = 256
  n_fft: int = 1024
  f_min: int = 0
  f_max: int = 8000
  n_mels: int = 80
  power: float = 1.0
  pad_value: float = -11.5129251  # value of melspectrograms if we fed a silence into `MelSpectrogram`


class MelSpectrogram(nn.Module):

  def __init__(self, config: MelSpectrogramConfig):
    super(MelSpectrogram, self).__init__()
    self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.sr, win_length=config.win_length, hop_length=config.hop_length,
                                                                n_fft=config.n_fft, f_min=config.f_min, f_max=config.f_max, n_mels=config.n_mels)
    # The is no way to set power in constructor in 0.5.0 version
    self.mel_spectrogram.spectrogram.power = config.power
    # Default torchaudio mel basis uses HTK formula. In order to be compatible with WaveGlow, we decided to use Slaney one instead (as well as librosa does by default)
    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.n_fft, n_mels=config.n_mels, fmin=config.f_min, fmax=config.f_max).T
    self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
    self.hop_length = config.hop_length

  def forward(self, audio: torch.Tensor) -> torch.Tensor:
    """
    :param audio: Expected shape is [B, T]
    :return: Shape is [B, n_mels, T']
    """
    B, T = audio.shape
    return self.mel_spectrogram(audio).clamp_(min=1e-5).log_()[:, :, :T // self.hop_length]


class LJSpeechDataset(torch.utils.data.Dataset):

  def __init__(self, segment_size, data_dir=None, part=None, shuffle=True, text_encoder=None, *args, **kwargs):
    assert text_encoder is None, "This guy doesn't believe in text encoders"
    data_dir = data_dir or str(ROOT_PATH / "data" / "datasets" / "ljspeech")
    assert os.path.exists(data_dir), f"Please place the dataset at {data_dir}"
    assert part in ['train', 'valid']
    regexp = str(data_dir + ('/wavs/LJ0[0,1,2,3,5]*.wav' if part == 'train' else '/wavs/LJ04*.wav'))
    self.audio_files = glob.glob(regexp)
    assert part != 'train' or len(self.audio_files) == 10712
    assert part != 'valid' or len(self.audio_files) == 2388
    if shuffle:
      np.random.shuffle(self.audio_files)
    self.segment_size = segment_size
    self.mel_spectrogramer = MelSpectrogram(MelSpectrogramConfig)

  def __getitem__(self, index):
    path = self.audio_files[index]
    audio = torchaudio.load(path)[0]  # (1, T)
    assert audio.ndim == 2, f'Got audio shape {audio.shape} in file {path}, expected (1, T)'
    _, T = audio.shape
    assert _ == 1, f'Got audio shape {audio.shape} in file {path}, expected (1, T)'
    if self.segment_size is not None:  # "if self.split" in the original code
      if self.segment_size <= T:
        start = np.random.randint(T - self.segment_size)
        audio = audio[:, start:start + self.segment_size]
      else:
        audio = F.pad(audio, (0, self.segment_size - T))
    mel_spectrogram = self.mel_spectrogramer(audio)[0]
    assert audio.ndim == 2, f'{audio.shape=}, expected (1, ..)'
    assert mel_spectrogram.ndim == 2, f'{mel_spectrogram.shape=}, expected (80, ..)'
    #print(f'{audio.shape=}, {mel_spectrogram.shape=}')
    return {
      "mel": mel_spectrogram,  # (80, ..)
      "audio": audio,          # (1, T)
    }

  def __len__(self):
    return len(self.audio_files)


class VocoderTestDataset(torch.utils.data.Dataset):

  def __init__(self, data_dir, *args, **kwargs):
    assert os.path.exists(data_dir), f"Please place the test dataset at {data_dir}"
    self.audio_files = list(pathlib.Path(data_dir).rglob('*.wav'))
    assert len(self.audio_files) > 0, f"No audio files found in the test dataset at {data_dir}"
    if len(self.audio_files) != 3:
      print('WARNING! This code was only tested on the three wav files given, but the test dataset that is detected is different. FUCK-UP EXPECTED!')
      agree_str = 'YES, I TAKE THE RESPONSIBILITY FOR ALL CRINGE AND PSYCHOLOGICAL ISSUES THAT WILL BE CAUSED'
      print(f'Are you sure you want to proceed? Please answer "{agree_str}"')
      if input() != agree_str:
        print('Different message. Aborting.')
        quit(2)
      print('Please confirm again, typing the same string')
      t0 = time()
      if input() != agree_str:
        print('Different message. Aborting.')
        quit(2)
      if time() - t0 < 8.00:
        print('You could not fully understand what you agree to in that short period of time( For the sake of your safety, you cannot proceed')
        quit(2)
      print('As you wish...')
    self.mel_spectrogramer = MelSpectrogram(MelSpectrogramConfig)

  def __getitem__(self, index):
    path = self.audio_files[index]
    audio = torchaudio.load(path)[0]  # (1, T)
    assert audio.ndim == 2, f'Got audio shape {audio.shape} in file {path}, expected (1, T)'
    _, T = audio.shape
    assert _ == 1, f'Got audio shape {audio.shape} in file {path}, expected (1, T)'
    mel_spectrogram = self.mel_spectrogramer(audio)[0]
    assert audio.ndim == 2, f'{audio.shape=}, expected (1, ..)'
    assert mel_spectrogram.ndim == 2, f'{mel_spectrogram.shape=}, expected (80, ..)'
    #print(f'{audio.shape=}, {mel_spectrogram.shape=}')
    return {
      "mel": mel_spectrogram,  # (80, ..)
      "audio": audio,          # (1, T)
      "save_to_path": 'generated-' + path.name
    }

  def __len__(self):
    return len(self.audio_files)
