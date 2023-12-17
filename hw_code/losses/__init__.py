from .ctc_loss_wrapper import CTCLossWrapper as CTCLoss
from .hifi_gan_losses import total_loss as hifi_gan_loss
from .asvspoof_losses import loss as asvspoof_loss

__all__ = [
  "CTCLoss",
  "hifi_gan_loss",
  "asvspoof_loss",
]
