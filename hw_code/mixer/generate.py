import os
from hw_code.mixer.mixture_generator import LibrispeechSpeaker, MixtureGenerator


#class LibrispeechSpeaker:
#  def __init__(self, all_audios_dir, speaker_id, mask='*.flac'):
#    self.id = int(speaker_id)
#    self.files = []
#    speaker_dir = os.path.join(all_audios_dir, str(speaker_id))
#    for chapter_dir in os.scandir(speaker_dir):
#      self.files += list(glob(os.path.join(speaker_dir, chapter_dir) + '/' + mask))
#    print('speaker', speaker_id, 'files', self.files)
#
#
#def snr_mixer(clean, noise, snr):
#  target_amp_noise = np.linalg.norm(clean) / 10 ** (snr / 20)
#  return clean + noise / np.linalg.norm(noise) * target_amp_noise
#
#
#def cut_audios(s1, s2, dur_sex, sr):
#  cut_len = dur_sex * sr
#  s1_cut, s2_cut = [], []
#  i = 0
#  while i * cut_len < len(s1) and (i + 1) * cut_len < len(s2):
#    s1_cut.append(s1[i * cut_len:(i + 1) * cut_len])
#    s2_cut.append(s2[i * cut_len:(i + 1) * cut_len])
#    i += 1
#  return s1_cut, s2_cut
#
#
#def mix(idx, triplet, out_folder, is_test, sr=16000, snr=0):
#  meter = pyln.Meter(sr)
#  normalize = lambda audio, db : pyln.normalize.loudness(audio, meter.integrated_loudness(audio), db)
#  s1, _ = sf.read(os.path.join('', triplet['target']))
#  s2, _ = sf.read(os.path.join('', triplet['noise']))
#  ref, _ = sf.read(os.path.join('', triplet['reference']))
#  s1, s2, ref = normalize(s1, -29), normalize(s2, -29), normalize(ref, -23)
#  path_mix = os.path.join(out_folder, f'{target_id}_{noise_id}_' + '%06d' % idx + '-mixed.wav')
#  path_target = os.path.join(out_folder, f'{target_id}_{noise_id}_' + '%06d' % idx + '-target.wav')
#  path_ref = os.path.join(out_folder, f'{target_id}_{noise_id}_' + '%06d' % idx + '-ref.wav')
#  if not is_test:
#    vad_merge = lambda audio : [audio[s:e] for s, e in librosa.effects.split(audio, top_db=vad_db)]
#    s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
#    s1_cut, s2_cut = cut_audios(s1, s2, audio_len, sr)
#    for i in range(len(s1_cut)):
#      mix = snr_mixer(s1_cut[i], s2_cut[i], snr)
#      sf.write(path_mix.replace('-mixed.wav', f'_{i}-mixed.wav'), normalize(mix, -23), sr)
#      sf.write(path_target.replace('-target.wav', f'_{i}-target.wav'), normalize(s1_cut[i], -23), sr)
#      sf.write(path_ref.replace('-ref.wav', f'_{i}-ref.wav'), ref, sr)
#  else:
#    s1, s2 = fix_length(s1, s2, 'max')
#    mix = snr_mixer(s1, s2, snr)
#    sf.write(path_mix, normalize(mix, -23), sr)
#    sf.write(path_target, normalize(s1, -23), sr)
#    sf.write(path_ref, ref, sr)
#
#
#class MixtureGenerator:
#  def __init__(self, speakers):
#    self.speakers = speakers
#
#  def sample_triplets(self, n_files):
#    all_triplets = {'reference': [], 'target': [], 'noise': [], 'target_sp_id': [], 'noise_sp_id': []}
#    while len(all_triples['reference']) < n_files:
#      speaker1, speaker2 = random.sample(self.speakers, 2)
#      if len(speaker1) < 2 or len(speaker2) < 2:
#        continue
#      target, reference = random.sample(speaker1.files, 1)
#      noise = random.choice(speaker2.files)
#      all_triples['reference'].append(reference)
#      all_triples['target'].append(target)
#      all_triples['noise'].append(noise)
#      all_triples['target_sp_id'].append(speaker1.id)
#      all_triples['noise_sp_id'].append(speaker2.id)
#    return all_triplets
#
#  def sample_triplets_conditional(self, n_files, target_speaker, noise_speaker):
#    n_files = min(n_files, len(target_speaker.files), len(noise_speaker.files))
#    return {
#      'reference': random.sample(target_speaker.files, n_files),
#      'target': random.sample(target_speaker.files, n_files),
#      'noise': random.sample(noise_speaker.files, n_files),
#      'target_sp_id': [target_speaker.id] * n_files,
#      'noise_sp_id': [noise_speaker.id] * n_files,
#    }
#
#  def generate_mixes(self, n_files, out_folder, update_interval=1000):
#    if not os.path.exists(out_folder):
#      os.makedirs(out_folder)
#    triplets = self.sample_triplets(n_files)
#    with ProcessPoolExecutor() as pool:
#      futures = []
#      for i in range(n_files):
#        triplet = {
#          'reference': triplets['reference'][i],
#          'target': triplets['target'][i],
#          'noise': triplets['noise'][i],
#          'target_sp_id': triplets['target_sp_id'][i],
#          'noise_sp_id': triplets['noise_sp_id'][i],
#        }
#        futures.append(pool.submit(mix, i, triplet, out_folder))
#      for i, future in enumerate(futures):
#        future.result()
#        print(f'Files processed: 0 out of {n_files}')
#        if (i + 1) % update_interval == 0:
#          print(f'Files processed: {i + 1} out of {n_files}')
#        print(f'Files processed: {n_files} out of {n_files}')


def generate(all_audios_dir, out_folder, n_files_train, n_files_test):
  all_speakers = [LibrispeechSpeaker(all_audios_dir, int(speaker_id)) for speaker_id in os.listdir(all_audios_dir)]
  mixer = MixtureGenerator(all_speakers)
  print('Generating Train')
  mixer.generate_mixes(n_files_train, os.path.join(out_folder, 'train'), is_test=False)
  print('Generating Test')
  mixer.generate_mixes(n_files_test, os.path.join(out_folder, 'test'), is_test=True)
