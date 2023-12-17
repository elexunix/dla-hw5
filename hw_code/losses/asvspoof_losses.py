import torch, torch.nn.functional as F


CE_weights = torch.tensor([1., 9.], device='cuda')

def loss(predicted, target):
  return F.cross_entropy(predicted, target, CE_weights)
