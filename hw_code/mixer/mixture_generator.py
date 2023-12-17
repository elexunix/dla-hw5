import numpy as np
import os
from shutil import rmtree
from glob import glob
import random
from concurrent.futures import ProcessPoolExecutor
import librosa, pyloudnorm as pyln, soundfile as sf


class LibrispeechSpeaker:
  def __init__(self, all_audios_dir, speaker_id, mask='*.flac'):
    self.id = int(speaker_id)
    self.files = []
    speaker_dir = os.path.join(all_audios_dir, str(speaker_id))
    for chapter_dir in os.listdir(speaker_dir):
      self.files += list(glob(os.path.join(speaker_dir, chapter_dir) + '/' + mask))
    print('speaker', speaker_id, len(self.files), 'files')


def snr_mixer(clean, noise, snr):
  target_amp_noise = np.linalg.norm(clean) / 10 ** (snr / 20)
  return clean + noise / np.linalg.norm(noise) * target_amp_noise


def cut_audios(s1, s2, dur_sex, sr):
  cut_len = dur_sex * sr
  s1, s2 = fix_length(s1, s2, 'max')
  s1_cut, s2_cut = [], []
  i = 0
  while (i + 1) * cut_len < len(s1):
    s1_cut.append(s1[i * cut_len:(i + 1) * cut_len])
    s2_cut.append(s2[i * cut_len:(i + 1) * cut_len])
    i += 1
  return s1_cut, s2_cut


def fix_length(s1, s2, min_or_max):
  if min_or_max == 'min':
    new_len = min(len(s1), len(s2))
    return s1[:new_len], s2[:new_len]
  else:
    new_len = max(len(s1), len(s2))
    return np.append(s1, np.zeros(new_len - len(s1))), np.append(s2, np.zeros(new_len - len(s2)))


def mix(idx, triplet, out_folder, is_test, sr=16000, snr=0):
  meter = pyln.Meter(sr)
  normalize = lambda audio, db : pyln.normalize.loudness(audio, meter.integrated_loudness(audio), db)
  s1, _ = sf.read(os.path.join('', triplet['target']))
  s2, _ = sf.read(os.path.join('', triplet['noise']))
  ref, _ = sf.read(os.path.join('', triplet['reference']))
  target_sp_id, noise_sp_id = triplet['target_sp_id'], triplet['noise_sp_id']
  s1, s2, ref = normalize(s1, -29), normalize(s2, -29), normalize(ref, -23)
  path_mix = os.path.join(out_folder, f'{target_sp_id}_{noise_sp_id}_' + '%06d' % idx + '-mixed.wav')
  path_target = os.path.join(out_folder, f'{target_sp_id}_{noise_sp_id}_' + '%06d' % idx + '-target.wav')
  path_ref = os.path.join(out_folder, f'{target_sp_id}_{noise_sp_id}_' + '%06d' % idx + '-ref.wav')
  if not is_test:
    #vad_merge = lambda audio : [audio[s:e] for s, e in librosa.effects.split(audio)]
    #s1, s2 = vad_merge(s1), vad_merge(s2)
    audio_len = 10
    s1_cut, s2_cut = cut_audios(s1, s2, audio_len, sr)
    for i in range(len(s1_cut)):
      mix = snr_mixer(s1_cut[i], s2_cut[i], snr)
      if np.isnan(normalize(mix, -23)).any() or np.isnan(normalize(s1_cut[i], -23)).any() or np.isnan(ref).any():
        continue
      sf.write(path_mix.replace('-mixed.wav', f'_{i}-mixed.wav'), normalize(mix, -23), sr)
      sf.write(path_target.replace('-target.wav', f'_{i}-target.wav'), normalize(s1_cut[i], -23), sr)
      sf.write(path_ref.replace('-ref.wav', f'_{i}-ref.wav'), ref, sr)
  else:
    s1, s2 = fix_length(s1, s2, 'max')
    mix = snr_mixer(s1, s2, snr)
    if np.isnan(normalize(mix, -23)).any() or np.isnan(normalize(s1, -23)).any() or np.isnan(ref).any():
      return
    sf.write(path_mix, normalize(mix, -23), sr)
    sf.write(path_target, normalize(s1, -23), sr)
    sf.write(path_ref, ref, sr)


class MixtureGenerator:
  def __init__(self, speakers):
    self.speakers = speakers

  def sample_triplets(self, n_files):
    all_triplets = {'reference': [], 'target': [], 'noise': [], 'target_sp_id': [], 'noise_sp_id': []}
    while len(all_triplets['reference']) < n_files:
      speaker1, speaker2 = random.sample(self.speakers, 2)
      if len(speaker1.files) < 2 or len(speaker2.files) < 2:
        continue
      target, reference = random.sample(speaker1.files, 2)
      noise = random.choice(speaker2.files)
      all_triplets['reference'].append(reference)
      all_triplets['target'].append(target)
      all_triplets['noise'].append(noise)
      all_triplets['target_sp_id'].append(speaker1.id)
      all_triplets['noise_sp_id'].append(speaker2.id)
    return all_triplets

  def sample_triplets_conditional(self, n_files, target_speaker, noise_speaker):
    n_files = min(n_files, len(target_speaker.files), len(noise_speaker.files))
    return {
      'reference': random.sample(target_speaker.files, n_files),
      'target': random.sample(target_speaker.files, n_files),
      'noise': random.sample(noise_speaker.files, n_files),
      'target_sp_id': [target_speaker.id] * n_files,
      'noise_sp_id': [noise_speaker.id] * n_files,
    }

  def generate_mixes(self, n_files, out_folder, is_test, update_interval=1000):
    if os.path.exists(out_folder):
      rmtree(out_folder)
    if not os.path.exists(out_folder):
      os.makedirs(out_folder)
    triplets = self.sample_triplets(n_files)
    with ProcessPoolExecutor() as pool:
      futures = []
      for i in range(n_files):
        triplet = {
          'reference': triplets['reference'][i],
          'target': triplets['target'][i],
          'noise': triplets['noise'][i],
          'target_sp_id': triplets['target_sp_id'][i],
          'noise_sp_id': triplets['noise_sp_id'][i],
        }
        futures.append(pool.submit(mix, i, triplet, out_folder, is_test))
      print(f'Files processed: 0 out of {n_files}')
      for i, future in enumerate(futures):
        future.result()
        if (i + 1) % update_interval == 0:
          print(f'Files processed: {i + 1} out of {n_files}')
      print(f'All {n_files} files processed')
