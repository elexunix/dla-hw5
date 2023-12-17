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


class NVTrainer(BaseTrainer):
  """
  Neural Vocoder Trainer class
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
    self.metrics_from_total_loss_names = ['total_loss', 'loss_gen1', 'loss_gen2', 'loss_disc1', 'loss_disc2', 'loss_mel', 'loss_fm1', 'loss_fm2',
                                          'mean_disc1_on_real', 'mean_disc1_on_fake', 'mean_disc2_on_real', 'mean_disc2_on_fake']
    self.train_metrics = MetricTracker(*self.metrics_from_total_loss_names, 'grad_norm', *[m.name for m in self.metrics], writer=self.writer)
    self.evaluation_metrics = MetricTracker(*self.metrics_from_total_loss_names, *[m.name for m in self.metrics], writer=self.writer)

  @staticmethod
  def move_batch_to_device(batch, device: torch.device):
    assert sorted(batch.keys()) in [sorted(['mel', 'mel_lens', 'audio', 'audio_lens']), sorted(['mel', 'mel_lens', 'audio', 'audio_lens', 'save_to_path'])], \
        f'Batch has keys {batch.keys()}, not recognized...'
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
      self.train_metrics.update('grad_norm', self.get_grad_norm())
      if batch_idx % self.log_step == 0:
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {batch["total_loss"]:.6f}')
        self.writer.add_scalar('learning rate', self.lr_scheduler.get_last_lr()[0])
        #self._log_predictions(**batch)
        self._log_spectrogram('sample real mel-spectrogram', batch['real_mel'])
        self._log_spectrogram('sample fake mel-spectrogram', batch['fake_mel'])
        self._log_spectrogram('sample real mel-spectrogram', batch['real_mel'])
        self._log_spectrogram('sample fake mel-spectrogram', batch['fake_mel'])
        self._log_audio('real audio', batch['real_audio'])
        self._log_audio('fake audio', batch['fake_audio'])
        self._log_scalars(self.train_metrics)
        # we don't want to reset train metrics at the start of every epoch
        # because we are interested in recent train metrics
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()
      if batch_idx >= self.len_epoch:
        break
    log = last_train_metrics

    for part, dataloader in self.evaluation_dataloaders.items():
      val_log = self._evaluation_epoch(epoch, part, dataloader)
      log.update(**{f'{part}_{name}': value for name, value in val_log.items()})

    return log

  def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
    batch = self.move_batch_to_device(batch, self.device)
    if is_train:
      self.optimizer.zero_grad()
    total_loss, output = self.model(batch['audio'], batch['mel'])
    if type(output) is dict:
      batch.update(output)
    else:
      assert False
      #batch['logits'] = output
    #batch['log_probs'] = F.log_softmax(batch['logits'], dim=-1)
    #batch['log_probs_length'] = self.model.transform_input_lengths(batch['spectrogram_length'])
    #batch.update(self.compute_losses(*outputs[:3], batch['target'], outputs[3], batch['target_sp_id']))
    if is_train:
      total_loss.backward()
      self._clip_grad_norm()
      self.optimizer.step()
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    # discriminators features are not reported
    for loss_name in self.metrics_from_total_loss_names:
      metrics.update(loss_name, output[loss_name])
    for met in self.metrics:
      metrics.update(met.name, met(**batch))
    return output

  def _evaluation_epoch(self, epoch, part, dataloader):
    """
    Validate after training an epoch

    :param epoch: integer
    :return: a log that contains information about validation
    """
    self.model.eval()
    self.evaluation_metrics.reset()
    with torch.no_grad():
      for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
        batch = self.process_batch(batch, is_train=False, metrics=self.evaluation_metrics)
      self.writer.set_step(epoch * self.len_epoch, part)
      self._log_scalars(self.evaluation_metrics)
      #self._log_predictions(**batch)
      self._log_spectrogram('real mel-spectrogram', batch['real_mel'])
      self._log_spectrogram('fake mel-spectrogram', batch['fake_mel'])
      self._log_audio('real audio', batch['real_audio'])
      self._log_audio('fake audio', batch['fake_audio'])
    # don't add histogram of model parameters to the tensorboard
    #for name, p in self.model.named_parameters():
    #  self.writer.add_histogram(name, p, bins='auto')
    return self.evaluation_metrics.result()

  def _progress(self, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(self.train_dataloader, 'n_samples'):
      current, total = batch_idx * self.train_dataloader.batch_size, self.train_dataloader.n_samples
    else:
      current, total = batch_idx, self.len_epoch
    return base.format(current, total, 100.0 * current / total)

#  def _log_predictions(self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, *args, **kwargs):
#    if self.writer is None:
#      return
#    argmax_inds = log_probs.cpu().argmax(-1).numpy()
#    argmax_inds = [inds[: int(ind_len)] for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())]
#    argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
#    argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
#    tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))
#    random.shuffle(tuples)
#    rows = {}
#    for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
#      target = BaseTextEncoder.normalize_text(target)
#      wer = calc_wer(target, pred) * 100
#      cer = calc_cer(target, pred) * 100
#      rows[Path(audio_path).name] = {
#        'target': target,
#        'raw prediction': raw_pred,
#        'predictions': pred,
#        'wer': wer,
#        'cer': cer,
#      }
#    self.writer.add_table('predictions', pd.DataFrame.from_dict(rows, orient='index'))

  def _log_spectrogram(self, caption, spectrogram_batch):
    spectrogram = random.choice(spectrogram_batch.cpu())
    image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
    self.writer.add_image(caption, torchvision.transforms.ToTensor()(image))

  def _log_audio(self, caption, audio_batch):
    audio = random.choice(audio_batch.cpu())
    self.writer.add_audio(caption, audio, sample_rate=22050)

  @torch.no_grad()
  def get_grad_norm(self, norm_type=2):
    parameters = self.model.parameters()
    if isinstance(parameters, torch.Tensor):
      parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]), norm_type)
    return total_norm.item()

  def _log_scalars(self, metric_tracker: MetricTracker):
    if self.writer is None:
      return
    for metric_name in metric_tracker.keys():
      self.writer.add_scalar(f'{metric_name}', metric_tracker.avg(metric_name))

  def mask_length(self, xs, lengths):
    assert len(xs) == len(lengths)
    result = torch.zeros_like(xs)
    for i, l in enumerate(lengths):
      result[i, :l] = xs[i, :l]
    return result

  #def compute_losses(self, pred_short, pred_middle, pred_long, target, speaker_logits, target_sp_id):
  #  #self.mask_length(pred_short, 
  #  sisdr_short, sisdr_middle, sisdr_long = self.sisdr(pred_short, target).mean(), self.sisdr(pred_middle, target).mean(), self.sisdr(pred_long, target).mean()
  #  ce_loss = F.cross_entropy(speaker_logits, target_sp_id)
  #  clf_accuracy = (speaker_logits.argmax(1) == target_sp_id).mean()
  #  return -.8 * sisdr_short - .1 * sisdr_middle - .1 * sisdr_long + 10 * ce_loss, clf_accuracy
