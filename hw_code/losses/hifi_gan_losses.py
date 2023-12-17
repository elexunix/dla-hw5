import torch, torch.nn.functional as F


def features_loss(features_on_real, features_on_fake):
  loss = 0
  for disc_output_on_real, disc_output_on_fake in zip(features_on_real, features_on_fake):
    for real, fake in zip(disc_output_on_real, disc_output_on_fake):
      loss += (real - fake).abs().mean()
  return loss


def generator_loss(disc_output_on_fake):
  losses = []
  for o in disc_output_on_fake:
    losses.append(((1 - o) ** 2).mean())
  return sum(losses), losses


def discriminator_loss(disc_output_on_real, disc_output_on_fake):
  losses_on_real, losses_on_fake = [], []
  for o_real, o_fake in zip(disc_output_on_real, disc_output_on_fake):
    losses_on_real.append(((1 - o_real) ** 2).mean())
    losses_on_fake.append((o_fake ** 2).mean())
  return sum(losses_on_real) + sum(losses_on_fake), losses_on_real, losses_on_fake


def mel_loss(real_mel, fake_mel):  # as I understand, this is used to ensure that we really generate a waveform for our given mel, not just a random realistic waveform
  return F.l1_loss(real_mel, fake_mel)


def total_loss(real_audio, fake_audio, real_mel, fake_mel, disc1_output_on_real_audio__G, disc1_features_on_real_audio__G, disc2_output_on_real_audio__G, disc2_features_on_real_audio__G,
                                                           disc1_output_on_real_audio__D, disc1_features_on_real_audio__D, disc2_output_on_real_audio__D, disc2_features_on_real_audio__D,
                                                           disc1_output_on_fake_audio__G, disc1_features_on_fake_audio__G, disc2_output_on_fake_audio__G, disc2_features_on_fake_audio__G,
                                                           disc1_output_on_fake_audio__D, disc1_features_on_fake_audio__D, disc2_output_on_fake_audio__D, disc2_features_on_fake_audio__D):
  loss_gen1, _ = generator_loss(disc1_output_on_fake_audio__G)                                         # part of adversarial loss
  loss_gen2, _ = generator_loss(disc2_output_on_fake_audio__G)                                         # part of adversarial loss
  loss_disc1, _, _ = discriminator_loss(disc1_output_on_real_audio__D, disc1_output_on_fake_audio__D)  # part of adversarial loss
  loss_disc2, _, _ = discriminator_loss(disc2_output_on_real_audio__D, disc2_output_on_fake_audio__D)  # part of adversarial loss
  loss_mel = mel_loss(real_mel, fake_mel)                                                              # L1 mel-spectrogram Loss
  loss_fm1 = features_loss(disc1_features_on_real_audio__G, disc1_features_on_fake_audio__G)           # part of feature matching loss
  loss_fm2 = features_loss(disc2_features_on_real_audio__G, disc2_features_on_fake_audio__G)           # part of feature matching loss
  #loss_total = 45 * loss_mel  # yeah, it works! mel 0.3
  loss_total = loss_gen1 + loss_gen2 + loss_disc1 + loss_disc2 + 2 * loss_fm1 + 2 * loss_fm2 + 45 * loss_mel
  mean = lambda l : sum(e.mean() for e in l).item() / len(l)
  ret = loss_total, {
    'total_loss': loss_total.item(),
    'loss_gen1': loss_gen1.item(),
    'loss_gen2': loss_gen2.item(),
    'loss_disc1': loss_disc1.item(),
    'loss_disc2': loss_disc2.item(),
    'loss_mel': loss_mel.item(),
    'loss_fm1': loss_fm1.item(),
    'loss_fm2': loss_fm2.item(),
    'real_audio': real_audio,
    'fake_audio': fake_audio.detach(),
    'real_mel': real_mel,
    'fake_mel': fake_mel.detach(),
    'mean_disc1_on_real': mean(disc1_output_on_real_audio__D),
    'mean_disc1_on_fake': mean(disc1_output_on_fake_audio__D),
    'mean_disc2_on_real': mean(disc2_output_on_real_audio__D),
    'mean_disc2_on_fake': mean(disc2_output_on_fake_audio__D),
    # discriminators features are not reported
  }
  return ret
