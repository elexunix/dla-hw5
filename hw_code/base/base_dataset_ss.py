import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_code.base.base_text_encoder import BaseTextEncoder
from hw_code.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDatasetSS(Dataset):
  def __init__(
      self,
      index,
      text_encoder: BaseTextEncoder,
      config_parser: ConfigParser,
      wave_augs=None,
      spec_augs=None,
      limit=None,
  ):
    self.text_encoder = text_encoder
    self.config_parser = config_parser
    self.wave_augs = wave_augs
    self.spec_augs = spec_augs
    self.log_spec = config_parser['preprocessing']['log_spec']
    self._assert_index_is_valid(index)
    self._index: List[dict] = index

  def __getitem__(self, ind):
    data_dict = self._index[ind]
    path_mixed, path_target, path_ref = data_dict['path_mixed'], data_dict['path_target'], data_dict['path_ref']
    wave_mixed, spec_mixed = self.process_wave(self.load_audio(path_mixed))
    wave_target, spec_target = self.process_wave(self.load_audio(path_target))
    wave_ref, spec_ref = self.process_wave(self.load_audio(path_ref))
    print(target_path.stem.split('_')[0])
    return {
      'mixed': spec_mixed,
      'target': spec_target,
      'ref': spec_ref,
      #'duration': audio_wave.size(1) / self.config_parser['preprocessing']['sr'],
      #'text_encoded': self.text_encoder.encode(data_dict['text']),
    }

  def __len__(self):
    return len(self._index)

  def load_audio(self, path):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :] # remove all channels but the first
    target_sr = self.config_parser['preprocessing']['sr']
    if sr != target_sr:
      audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor

  def process_wave(self, audio_tensor_wave: Tensor):
    with torch.no_grad():
      if self.wave_augs is not None:
        audio_tensor_wave = self.wave_augs(audio_tensor_wave)
      wave2spec = self.config_parser.init_obj(self.config_parser['preprocessing']['spectrogram'], torchaudio.transforms)
      audio_tensor_spec = wave2spec(audio_tensor_wave)
      if self.spec_augs is not None:
        audio_tensor_spec = self.spec_augs(audio_tensor_spec)
      if self.log_spec:
        audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
      return audio_tensor_wave, audio_tensor_spec

  @staticmethod
  def _assert_index_is_valid(index):
    for entry in index:
      assert 'path_mixed' in entry, 'Each dataset item should include field "path_mixed" - path to mixed audio file.'
      assert 'path_target' in entry, 'Each dataset item should include field "path_target" - path to target audio file.'
      assert 'path_ref' in entry, 'Each dataset item should include field "path_ref" - path to reference speech audio file.'
