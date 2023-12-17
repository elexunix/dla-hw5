import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_code.base.base_dataset_ss import BaseDatasetSS
from hw_code.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class SourceSeparationDataset(BaseDatasetSS):
  def __init__(self, part, data_dir=None, *args, **kwargs):
    assert part in ['train', 'test']
    if data_dir is None:
      data_dir = ROOT_PATH / 'data' / 'datasets' / 'source-separation' / part
    self.data_dir = data_dir
    self.index = [{
      'path_mixed': str(index_mixed_entry),
      'path_target': str(index_mixed_entry)[:-10] + '-target.wav',
      'path_ref': str(index_mixed_entry)[:-10] + '-ref.wav',
      'target_sp_id': int(str(index_mixed_entry).split('/')[-1].split('_')[0]),
    } for index_mixed_entry in data_dir.glob('*-mixed.wav')]
    assert len(self.index) > 0, f'Dataset not found: no files matched {data_dir.resolve()}/*-mixed.wav'
    super().__init__(self.index, *args, **kwargs)

  def __getitem__(self, index):
    return {
      'mixed': self.load_audio(self.index[index]['path_mixed']),
      'target': self.load_audio(self.index[index]['path_target']),
      'ref': self.load_audio(self.index[index]['path_ref']),
      'target_sp_id': self.index[index]['target_sp_id'],
    }

  def __len__(self):
    return len(self.index)
