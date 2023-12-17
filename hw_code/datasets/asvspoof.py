import os, pathlib, glob
from dataclasses import dataclass
import random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, librosa
from hw_code.utils import ROOT_PATH

# format samples:
#LA_0022 LA_E_6977360 - A09 spoof
#LA_0031 LA_E_5932896 - A13 spoof
#LA_0030 LA_E_5849185 - - bonafide

class ASVSpoofDataset(torch.utils.data.Dataset):

  def __init__(self, segment_size, data_dir=None, part=None, max_size=None, shuffle=True, *args, **kwargs):
    data_dir = data_dir or str(ROOT_PATH / "data" / "datasets" / "asvspoof")
    assert os.path.exists(data_dir), f"Please place the dataset at {data_dir}"
    assert part in ["train", "valid", "test"]
    if part == "train":
      f_expr      = data_dir + "/LA/LA/ASVspoof2019_LA_train/flac/{}.flac"
      labels_file = data_dir + "/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    elif part == "valid":
      f_expr      = data_dir + "/LA/LA/ASVspoof2019_LA_dev/flac/{}.flac"
      labels_file = data_dir + "/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    elif part == "test":
      f_expr      = data_dir + "/LA/LA/ASVspoof2019_LA_eval/flac/{}.flac"
      labels_file = data_dir + "/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    self.audio_files = [f_expr.format(name.split()[1]) for name in open(labels_file).read().split('\n')[:-1]]
    assert part != "train" or len(self.audio_files) == 25380, "We condemn the use of other datasets"
    assert part != "valid" or len(self.audio_files) == 24844, "We condemn the use of other datasets"
    assert part != "test" or len(self.audio_files) == 71237, "We condemn the use of other datasets"
    self.target_labels = {}
    with open(labels_file) as file:
      for line in file:
        parts = line.split()
        self.target_labels[parts[1]] = parts[-1]
    self.target_labels = [0 if self.target_labels[path.split('/')[-1].split('.')[0]] == 'spoof' else 1 for path in self.audio_files]
    if shuffle:
      permutation = list(range(len(self.audio_files)))
      random.shuffle(permutation)
      self.audio_files = [self.audio_files[i] for i in permutation]
      self.target_labels = [self.target_labels[i] for i in permutation]
    self.segment_size = segment_size
    if max_size is not None:
      self.audio_files = self.audio_files[:max_size]
      self.target_labels = self.target_labels[:max_size]

  def __getitem__(self, index):
    path = self.audio_files[index]
    audio = torchaudio.load(path)[0]  # (1, T)
    audio = audio[:self.segment_size]
    audio = torch.cat([audio.tile((1, self.segment_size // audio.shape[-1])), audio[:, :self.segment_size % audio.shape[-1]]], 1)
    assert audio.shape == (1, self.segment_size), f"Audio has shape {audio.shape}, but (1, {self.segment_size}) is needed..."
    assert audio.ndim == 2, f"Got audio shape {audio.shape} in file {path}, expected (1, T)"
    assert audio.shape[0] == 1, f"Got audio shape {audio.shape} in file {path}, expected (1, T)"
    return {
      "audio": audio,  # (1, T)
      "label": self.target_labels[index],
    }

  def __len__(self):
    return len(self.audio_files)
