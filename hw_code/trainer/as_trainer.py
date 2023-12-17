import random
from pathlib import Path
import PIL
import pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

from hw_code.base import BaseTrainer
from hw_code.base.base_text_encoder import BaseTextEncoder
from hw_code.logger.utils import plot_spectrogram_to_buf
from hw_code.metric.utils import calc_cer, calc_wer
from hw_code.utils import inf_loop, MetricTracker
from hw_code.losses import asvspoof_loss
from calculate_eer import compute_eer


class ASTrainer(BaseTrainer):
  """
  Anti-Spoofing Trainer class
  """

  def __init__(self, model, metrics, optimizer, config, device, dataloaders, lr_scheduler=None, len_epoch=None, skip_oom=True):
    super().__init__(model, None, metrics, optimizer, config, device)
    self.skip_oom = skip_oom
    self.config = config
    self.train_dataloader = dataloaders['train']
    if len_epoch is None:  # epoch-based training
      self.len_epoch = len(self.train_dataloader)
    else:  # iteration-based training
      self.train_dataloader = inf_loop(self.train_dataloader)
      self.len_epoch = len_epoch
    self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}
    self.lr_scheduler = lr_scheduler
    self.log_step = self.config['trainer']['log_interval']
    self.train_metrics = MetricTracker('loss', 'grad_norm', *[m.name for m in self.metrics], writer=self.writer)
    self.validation_metrics = MetricTracker('loss', *[m.name for m in self.metrics], writer=self.writer)
    self.test_metrics = MetricTracker('loss', 'eer', *[m.name for m in self.metrics], writer=self.writer)

  @staticmethod
  def move_batch_to_device(batch, device: torch.device):
    assert sorted(batch.keys()) == sorted(['audio', 'label']), f'Batch has keys {batch.keys()}, not recognized...'
    for key in batch.keys():
      if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].to(device)
    return batch

  def _clip_grad_norm(self):
    if self.config['trainer'].get('grad_norm_clip', None) is not None:
      nn.utils.clip_grad_norm_(
        self.model.parameters(), self.config['trainer']['grad_norm_clip']
      )

  def _train_epoch(self, epoch):
    """
    Training logic for an epoch

    :param epoch: integer
    :return: a log that contains average loss and metric in this epoch
    """
    self.model.train()
    self.train_metrics.reset()
    self.writer.add_scalar('epoch', epoch)
    for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc='train', total=self.len_epoch)):
      try:
        batch = self.process_batch(batch, is_train=True, metrics=self.train_metrics)
      except RuntimeError as e:
        if 'out of memory' in str(e) and self.skip_oom:
          self.logger.warning('OOM on batch. Skipping batch.')
          for p in self.model.parameters():
            if p.grad is not None:
              del p.grad  # free some memory
          torch.cuda.empty_cache()
          continue
        else:
          raise e
      #self.train_metrics.update('loss', batch['loss'])
      self.train_metrics.update('grad_norm', self.get_grad_norm())
      if batch_idx % self.log_step == 0:
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {batch["loss"]:.6f}')
        self.writer.add_scalar('learning rate', self.lr_scheduler.get_last_lr()[0])
        self._log_scalars('train', self.train_metrics)
        # we don't want to reset train metrics at the start of every epoch
        # because we are interested in recent train metrics
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()
      if batch_idx >= self.len_epoch:
        break
    log = last_train_metrics

    for part, dataloader in self.evaluation_dataloaders.items():
      if part == 'valid':
        new_log = self._validation_epoch(epoch, part, dataloader)
      elif part == 'test':
        new_log = self._test_epoch(epoch, part, dataloader)
      else:
        assert False, f'Part not recognized: {part}'
      log.update(**{f'{part}_{name}': value for name, value in new_log.items()})

    return log

  def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
    batch = self.move_batch_to_device(batch, self.device)
    if is_train:
      self.optimizer.zero_grad()
    output, target = self.model(batch['audio']), batch['label']
    if type(output) is dict:
      assert False
      batch.update(output)
    else:
      batch['logits'] = output.detach().cpu()
    loss = asvspoof_loss(output, target)
    batch['loss'] = loss.item()
    #batch['log_probs'] = F.log_softmax(batch['logits'], dim=-1)
    #batch['log_probs_length'] = self.model.transform_input_lengths(batch['spectrogram_length'])
    #batch.update(self.compute_losses(*outputs[:3], batch['target'], outputs[3], batch['target_sp_id']))
    if is_train:
      loss.backward()
      self._clip_grad_norm()
      self.optimizer.step()
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    metrics.update('loss', batch['loss'])
    for met in self.metrics:
      assert False
      #print(batch, met, met.name, met(**batch))
      metrics.update(met.name, met(**batch))
    return batch

  def _validation_epoch(self, epoch, part, dataloader):
    """
    Validate after training an epoch

    :param epoch: integer
    :return: a log that contains information about validation
    """
    self.model.eval()
    self.validation_metrics.reset()
    with torch.no_grad():
      for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
        batch = self.process_batch(batch, is_train=False, metrics=self.validation_metrics)
    self.writer.set_step(epoch * self.len_epoch, part)
    self._log_scalars('valid', self.validation_metrics)
    return self.validation_metrics.result()

  def _test_epoch(self, epoch, part, dataloader):
    """
    Test after training an epoch

    :param epoch: integer
    :return: a log that contains information about test
    """
    self.model.eval()
    self.test_metrics.reset()
    logits_of_bonafide_on_bonafide, logits_of_bonafide_on_spoof = [], []
    with torch.no_grad():
      for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
        batch = self.move_batch_to_device(batch, self.device)
        logits, target = self.model(batch['audio']), batch['label']
        self.test_metrics.update('loss', asvspoof_loss(logits, target))
        logits_of_bonafide_on_bonafide += logits[target == 1, 1].cpu().tolist()
        logits_of_bonafide_on_spoof += logits[target == 0, 1].cpu().tolist()
    self.test_metrics.update('eer', compute_eer(logits_of_bonafide_on_bonafide, logits_of_bonafide_on_spoof)[0])
    self._log_scalars('test', self.test_metrics)
    return self.test_metrics.result()

  def _progress(self, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(self.train_dataloader, 'n_samples'):
      current, total = batch_idx * self.train_dataloader.batch_size, self.train_dataloader.n_samples
    else:
      current, total = batch_idx, self.len_epoch
    return base.format(current, total, 100.0 * current / total)

  #def _log_spectrogram(self, caption, spectrogram_batch):
  #  spectrogram = random.choice(spectrogram_batch.cpu())
  #  image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
  #  self.writer.add_image(caption, torchvision.transforms.ToTensor()(image))

  #def _log_audio(self, caption, audio_batch):
  #  audio = random.choice(audio_batch.cpu())
  #  self.writer.add_audio(caption, audio, sample_rate=16000)

  @torch.no_grad()
  def get_grad_norm(self, norm_type=2):
    parameters = self.model.parameters()
    if isinstance(parameters, torch.Tensor):
      parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]), norm_type)
    return total_norm.item()

  def _log_scalars(self, part: str, metric_tracker: MetricTracker):
    if self.writer is None:
      return
    for metric_name in metric_tracker.keys():
      self.writer.add_scalar(f'{metric_name}', metric_tracker.avg(metric_name))
