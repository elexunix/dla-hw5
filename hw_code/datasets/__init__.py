#from hw_code.datasets.custom_audio_dataset import CustomAudioDataset
#from hw_code.datasets.custom_dir_audio_dataset import CustomDirAudioDataset
from hw_code.datasets.librispeech import LibrispeechDataset
#from hw_code.datasets.ljspeech_dataset import LJspeechDataset
#from hw_code.datasets.common_voice import CommonVoiceDataset
from .source_separation import SourceSeparationDataset
from .ljspeech import LJSpeechDataset, VocoderTestDataset
MelDataset = LJSpeechDataset
from .asvspoof import ASVSpoofDataset

__all__ = [
  "LibrispeechDataset",
  #"CustomDirAudioDataset",
  #"CustomAudioDataset",
  #"LJspeechDataset",
  #"CommonVoiceDataset",
  "SourceSeparationDataset",
  "LJSpeechDataset",
  "MelDataset",
  "VocoderTestDataset",
  "ASVSpoofDataset",
]
