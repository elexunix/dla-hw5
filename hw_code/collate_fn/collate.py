import logging
from typing import Type, List
import torch, torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from hw_code.datasets import LibrispeechDataset, LJSpeechDataset, VocoderTestDataset, ASVSpoofDataset

logger = logging.getLogger(__name__)


def asr_collate_fn(dataset_items: List[dict]):
  """
  Collate and pad fields in dataset items
  """
  collated_batch = {}
  text_enc_lens, spec_lens = [], []
  for key, value0 in dataset_items[0].items():
    values = [item[key] for item in dataset_items]
    if isinstance(value0, torch.Tensor):
      lens = [value.shape[-1] for value in values]
      if key == 'text_encoded':
        text_enc_lens = lens
      elif key == 'spectrogram':
        spec_lens = lens
      collated_batch[key] = torch.vstack([F.pad(v, (0, max(lens) - v.shape[-1])) for v in values])
      device = value0.device
    else:
      collated_batch[key] = default_collate(values)
  collated_batch['text_encoded_length'] = torch.tensor(text_enc_lens, device=device)
  collated_batch['spectrogram_length'] = torch.tensor(spec_lens, device=device)
  #for k, v in collated_batch.items():
  #  print('key', k, 'value/shape', v.shape if isinstance(v, torch.Tensor) else v)

  return collated_batch


def nv_collate_fn(dataset_items: List[dict]):
  """
  Collate and pad fields in dataset items
  """
  collated_batch = {}
  for key, value0 in dataset_items[0].items():
    values = [item[key] for item in dataset_items]
    if isinstance(value0, torch.Tensor):
      lens = [value.shape[-1] for value in values]
      maxlen = max(lens)
      collated_batch[key] = torch.stack([F.pad(v, (0, maxlen - v.shape[-1])) for v in values])
      collated_batch[key + '_lens'] = torch.tensor(lens, device=value0.device)
    else:
      collated_batch[key] = default_collate(values)
  return collated_batch


def as_collate_fn(dataset_items: List[dict]):
  """
  Collate fields in dataset items, all should have same length
  """
  return default_collate(dataset_items)


def get_collate_fn(dataset):
  #print('GET_COLLATE_FN', type(dataset))
  if type(dataset) is LibrispeechDataset:  # hw 1, ASR
    return asr_collate_fn
  if type(dataset) is LJSpeechDataset or type(dataset) is VocoderTestDataset:  # hw 4, NV
    return nv_collate_fn
  if type(dataset) is ASVSpoofDataset:  # hw 5, AS
    return as_collate_fn
  assert False
