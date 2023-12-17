from typing import List

import torch
from torch import Tensor

from hw_code.base.base_metric import BaseMetric
from hw_code.base.base_text_encoder import BaseTextEncoder
from hw_code.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
  def __init__(self, text_encoder: BaseTextEncoder, name: str, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.text_encoder = text_encoder
    self.name = name
    self.cers = []

  def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
    predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
    lengths = log_probs_length.detach().numpy()
    cers = []
    for log_prob_vec, length, target_text in zip(predictions, lengths, text):
      target_text = BaseTextEncoder.normalize_text(target_text)
      if hasattr(self.text_encoder, "ctc_decode"):
        pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
      else:
        assert False
        pred_text = self.text_encoder.decode(log_prob_vec[:length])
      cers.append(calc_cer(target_text, pred_text))
    cer = sum(cers) / len(cers)
    self.cers.append(cer)
    return cer

  def toJSON(self):
    return {
      self.name: sum(self.cers) / len(self.cers)
    }


class BSCERMetric(BaseMetric):
  def __init__(self, text_encoder: BaseTextEncoder, use_lm: bool, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.text_encoder = text_encoder
    self.use_lm = use_lm
    self.cers = []

  def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
    predictions = log_probs.cpu().detach().numpy()
    lengths = log_probs_length.detach().numpy()
    cers = []
    for log_prob_vec, length, target_text in zip(predictions, lengths, text):
      target_text = BaseTextEncoder.normalize_text(target_text)
      if self.use_lm:
        pred_text = self.text_encoder.ctc_beam_search_with_lm(torch.tensor(log_prob_vec[:length]))
      else:
        pred_text = self.text_encoder.ctc_beam_search(log_prob_vec[:length])
      cers.append(calc_cer(target_text, pred_text))
    cer = sum(cers) / len(cers)
    self.cers.append(cer)
    return cer

  def toJSON(self):
    return {
      self.name: sum(self.cers) / len(self.cers)
    }
