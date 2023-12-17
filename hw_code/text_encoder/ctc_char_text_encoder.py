from typing import List, NamedTuple
import torch
from .char_text_encoder import CharTextEncoder
from tqdm import trange
import kenlm
import multiprocessing
from pyctcdecode import build_ctcdecoder
#import numpy as np
#from numba import jit


class Hypothesis(NamedTuple):
  text: str
  log_prob: float


class CTCCharTextEncoder(CharTextEncoder):
  EMPTY_TOK = "^"

  def __init__(self, alphabet: List[str]=None, lm_for_bs_path: str=None, librispeech_vocab_path: str=None):
    super().__init__(alphabet)
    vocab = [self.EMPTY_TOK] + list(self.alphabet)
    self.ind2char = dict(enumerate(vocab))
    self.char2ind = {v: k for k, v in self.ind2char.items()}
    if lm_for_bs_path is not None:
      #lm_for_bs = kenlm.Model(lm_for_bs_path)
      with open(librispeech_vocab_path) as f:
        unigram_list = [t.lower() for t in f.read().strip().split('\n')]
      self.bs_lm_object = build_ctcdecoder([''] + self.alphabet, lm_for_bs_path, unigram_list)

  def ctc_decode(self, inds: List[int]) -> str:
    last_char = self.EMPTY_TOK
    result = ''
    for i in inds:
      c = self.ind2char[i]
      if c != last_char:
        if c != self.EMPTY_TOK:
          result += c
        last_char = c
    return result

#  @jit
#  def ctc_beam_search_internal(self, log_probs: np.ndarray, length, beam_size: int = 3) -> List[Hypothesis]:
#    """
#    Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
#    """
#    log_probs = log_probs
#    assert log_probs.ndim == 2
#    char_length, voc_size = log_probs.shape
#    assert voc_size == 28
#    hypos: List[Hypothesis] = []
#
#    hypos = [Hypothesis(text='', prob=1.0)]
#    for t in range(length):
#      all_hypos = []
#      for h in hypos:
#        # Extend each hypothesis with all possible characters
#        for c in range(voc_size):
#          new_text = h.text + self.ind2char[c]
#          new_prob = h.prob + log_probs[t, c]
#          # Merge operation
#          if len(new_text) > 1 and new_text[-2:] == self.ind2char[c]*2:
#            new_text = new_text[:-1]
#          # Merge operation with empty token
#          if len(new_text) > 1 and new_text[-1] == self.EMPTY_TOK and new_text[-2] == self.ind2char[c]:
#            new_text = new_text[:-2] + self.ind2char[c]
#          all_hypos.append(Hypothesis(text=new_text, prob=new_prob))
#      # Keep only best hypotheses
#
#      log_probs_np = np.array([hypo[1] for hypo in all_hypos])
#      indices = log_probs_np.argsort()[::-1][:beam_size]
#      hypos = [all_hypos[i] for i in indices]
#
#    return hypos[0].text  #sorted(hypos, key=lambda x: x.prob, reverse=True)

  def ctc_beam_search(self, log_probs: torch.tensor, length, beam_size: int=3) -> List[Hypothesis]:
    assert False
#    return self.ctc_beam_search_internal(log_probs.cpu().detach().numpy(), length, beam_size)
    """
    Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
    """
    log_probs = log_probs.cpu().detach()
    assert len(log_probs.shape) == 2
    char_length, voc_size = log_probs.shape
    assert voc_size == len(self.ind2char)
    hypos: List[Hypothesis] = []

    hypos = [Hypothesis(text='', log_prob=0.)]
    for t in range(length):
      all_hypos = []
      for h in hypos:
        # Extend each hypothesis with all possible characters
        for c in range(voc_size):
          new_text = h.text + self.ind2char[c]
          new_log_prob = h.log_prob + log_probs[t, c]
          # Merge operation
          if len(new_text) > 1 and new_text[-2:] == self.ind2char[c]*2:
            new_text = new_text[:-1]
          # Merge operation with empty token
          if len(new_text) > 1 and new_text[-1] == self.EMPTY_TOK and new_text[-2] == self.ind2char[c]:
            new_text = new_text[:-2] + self.ind2char[c]
          all_hypos.append(Hypothesis(text=new_text, log_prob=new_log_prob))
      # Keep only best hypotheses
      hypos = sorted(all_hypos, key=lambda h: h.log_prob, reverse=True)[:beam_size]

    return hypos[0].text  #sorted(hypos, key=lambda x: x.log_prob, reverse=True)

  def ctc_beam_search_with_lm(self, log_probs: torch.tensor, beam_size: int=3) -> List[Hypothesis]:
    assert log_probs.ndim == 2
    return self.bs_lm_object.decode(log_probs.cpu().detach().numpy(), beam_width=beam_size)
