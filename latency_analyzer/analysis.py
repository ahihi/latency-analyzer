import librosa
import numpy as np
import sortednp

class ChannelAnalysis:
  def __init__(self, audio, sample_rate, onset_detect_kwargs={}):
    self.audio = audio
    self.sample_rate = sample_rate
    self.onsets = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, **onset_detect_kwargs)
    self.abs_max_amplitude = np.abs(self.audio).max()

class Analysis:
  def __init__(self, audio, sample_rate, onset_detect_kwargs={}, channels=None):
    # audio coming from librosa can have shape (num_channels, num_samples) or (num_samples,)
    is_1d = len(audio.shape) == 1
    self.num_channels = 1 if is_1d else audio.shape[0]

    if self.num_channels < 2:
      raise ValueError(f"expected at least 2 channels, got {self.num_channels}")
    
    self.audio = audio
    self.sample_rate = sample_rate
    self.sample_duration = 1.0 / sample_rate
    self.num_samples = audio.shape[1]
    self.duration = self.num_samples * self.sample_duration

    self.channel_indices = channels if channels is not None else range(self.num_channels)
    self.channels = [ChannelAnalysis(self.audio[i,:], self.sample_rate, onset_detect_kwargs) for i in self.channel_indices]
    self.onsets = sortednp.kway_merge(*(channel_analysis.onsets for channel_analysis in self.channels))
    
    self.abs_max_amplitude = max(self.channels, key=lambda ca: ca.abs_max_amplitude).abs_max_amplitude
