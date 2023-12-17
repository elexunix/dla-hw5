from typing import List

import torch
from torch import Tensor

from hw_code.base.base_metric import BaseMetric
from hw_code.base.base_text_encoder import BaseTextEncoder
from hw_code.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
  def __init__(self, text_encoder: BaseTextEncoder, name: str, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.text_encoder = text_encoder
    self.name = name
    self.wers = []

  def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
    predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
    lengths = log_probs_length.detach().numpy()
    wers = []
    for log_prob_vec, length, target_text in zip(predictions, lengths, text):
      target_text = BaseTextEncoder.normalize_text(target_text)
      if hasattr(self.text_encoder, "ctc_decode"):
        pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
      else:
        assert False
        pred_text = self.text_encoder.decode(log_prob_vec[:length])
      wers.append(calc_wer(target_text, pred_text))
    wer = sum(wers) / len(wers)
    self.wers.append(wer)
    return wer

  def toJSON(self):
    return {
      self.name: sum(self.wers) / len(self.wers)
    }


class BSWERMetric(BaseMetric):
  def __init__(self, text_encoder: BaseTextEncoder, use_lm: bool, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.text_encoder = text_encoder
    self.use_lm = use_lm
    self.wers = []

  def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
    predictions = log_probs.cpu().detach().numpy()
    lengths = log_probs_length.detach().numpy()
    wers = []
    for log_prob_vec, length, target_text in zip(predictions, lengths, text):
      target_text = BaseTextEncoder.normalize_text(target_text)
      if self.use_lm:
        pred_text = self.text_encoder.ctc_beam_search_with_lm(torch.tensor(log_prob_vec[:length]))
      else:
        pred_text = self.text_encoder.ctc_beam_search(log_prob_vec[:length])
      wers.append(calc_wer(target_text, pred_text))
    wer = sum(wers) / len(wers)
    self.wers.append(wer)
    return wer

  def toJSON(self):
    return {
      self.name: sum(self.wers) / len(self.wers)
    }
