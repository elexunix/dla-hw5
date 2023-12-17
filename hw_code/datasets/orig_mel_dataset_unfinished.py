import os
import numpy as np
import scipy
import torch, torch.nn.functional as F
import librosa


def load_wav(full_path):
  sampling_rate, data = scipy.io.wavfile.read(full_path)
  return data, sampling_rate


mel_bases = {}
hann_windows = {}

def mel_spectrogram(audio, n_fft, n_mels, sampling_rate, hop_size, window_size, fmin, fmax, center):
  if audio.min() < -1:
    print('audio.min() is', audio.min())
  if audio.max() > +1:
    print('audio.max() is', audio.max())
  global mel_bases
  mel_basis_name = str(fmax) + '_' + str(audio.device)
  if mel_basis_name not in mel_bases:
    mel_basis = librosa.filters.mel(sampling_rate, n_fft, n_mels, fmin, fmax)
    mel_bases[mel_basis_name] = torch.tensor(mel_basis, dtype=torch.float, device=audio.device)
    hann_window[str(audio.device)] = torch.hann_window(window_size, device=audio.device)
  audio = audio.unsqueeze(1)
  audio = F.pad(audio, ((n_fft - hop_size) // 2, (n_fft - hop_size) // 2), mode='reflect')
  audio = audio.squeeze(1)
  mel_spectrogram = torch.stft(x, n_fft, hop_size, window_size, hann_window[str(audio.device)], center)
  mel_spectrogram = torch.sqrt(mel_spectrogram.pow(2).sum(-1) + 1e-9)
  mel_spectrogram = mel_bases[mel_basis_name] @ mel_spectrogram
  return spectral_normalize_torch(mel_spectrogram)


class MelDataset(torch.utils.data.Dataset):

  def __init__(self, audio_files, segment_size, shuffle=True):
    self.audio_files = audio_files
    if shuffle:
      np.random.shuffle(self.audio_files)
    self.segment_size = segment_size

  def __getitem__(self, index):
    filename = self.audio_files[index]
    audio = torch.tensor(audio, dtype=torch.float)
    if self.segment_size:  # "if self.split" in original code
      if self.segment_size <= len(audio):
        start = np.random.randint(len(audio) - self.segment_size)
        audio = audio[start:start + segment_size]
      else:
        audio = F.pad(audio, (0, self.segment_size - len(audio)))
      mel = mel_spectrogram(audio[None], self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax, center=False)

