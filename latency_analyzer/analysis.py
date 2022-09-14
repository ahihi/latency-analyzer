from numbers import Number
import os
import time
from types import SimpleNamespace

import librosa
import numpy as np
import scipy
import sortednp

def peak_normalize(x):
  x = x - np.mean(x)
  x = x / np.max(np.abs(x))
  return x

def truncate_to_even(x, axis=-1):
  length = x.shape[axis]
  if length % 2 != 0:
    return np.take(x, range(length-1), axis=axis)
  else:
    return x

def envelope_rms(audio, win_len=2000):
  win_len_half = win_len // 2
  squared = np.power(audio, 2)
  padded = np.pad(squared, (win_len_half, win_len_half-1), "constant", constant_values=(0,0))
  window = np.ones(win_len) / win_len
  env = np.sqrt(np.convolve(padded, window, "valid"))

  # env[0] = 0.0
  # for i in range(1, win_len_half):
  #   env[i] = np.mean(audio[:i]**2)

  # for i in range(len(audio) - win_len_half + 1, len(audio)):
  #   env[i] = np.mean(audio[i - win_len_half : i + win_len_half]**2)
    
  env = peak_normalize(env)
  
  assert env.shape[0] == audio.shape[0], f"{env.shape[0]} != {audio.shape[0]}" 
  return env

def envelope_hilbert(x):
  mean = np.mean(x)
  x_centered = x - mean
  env = np.abs(scipy.signal.hilbert(x_centered))
  # env = np.abs(scipy.fftpack.hilbert(x_centered))
  env = env + mean
  assert env.shape[0] == x.shape[0], f"{env.shape[0]} != {x.shape[0]}"
  return env

# https://dsp.stackexchange.com/a/74822
def rolling_rms(x, n):
  x = np.concatenate(([0], x))
  xc = np.cumsum(abs(x)**2);
  return np.sqrt((xc[n:] - xc[:-n]) / n)

# https://stackoverflow.com/q/43652911
def xcorr_unbiased(x, y):
  assert x.size == y.size, f"x.size={x.size} != y.size={y.size}"
    
  corr = scipy.signal.correlate(x, y)
  # corr = np.correlate(x, y, "full") # takes forever

  lags = np.arange(-(x.size - 1), x.size)

  corr /= (x.size - np.abs(lags))
  
  return corr, lags

def duration_to_samples(duration, sample_rate):
  if isinstance(duration, Number):
    return duration
  
  sum = 0
  if hasattr(duration, "seconds"):
    sum += round(duration.seconds * sample_rate)
  if hasattr(duration, "samples"):
    sum += duration.samples
  return sum

class BonkChannelAnalysis:
  def __init__(self, audio, sample_rate, onset_detect_kwargs={}):
    self.audio = audio
    self.sample_rate = sample_rate
    self.onsets = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, **onset_detect_kwargs)
    self.abs_max_amplitude = np.abs(self.audio).max()

class BonkAnalysis:
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
    self.channels = [BonkChannelAnalysis(self.audio[i,:], self.sample_rate, onset_detect_kwargs) for i in self.channel_indices]
    self.onsets = sortednp.kway_merge(*(channel_analysis.onsets for channel_analysis in self.channels))
    
    self.abs_max_amplitude = max(self.channels, key=lambda ca: ca.abs_max_amplitude).abs_max_amplitude

class SwingAnalysis:
  win_types = {
    "rect": lambda n: np.ones(n, dtype=np.float32),
    "hann": lambda n: scipy.signal.windows.hann(n), # TODO: symmetric (default) or periodic?
  }
  default_win_type = "hann"
  
  def __init__(self, mic_sig, render_sig, sample_rate, rms_win_len, win_len=None, win_hop=None, win_type=None, swing_freq=None, path=None):
    self.path = path
    self.filename = os.path.basename(self.path) if self.path else "<no filename>"

    self.sample_rate = sample_rate
    self.sample_duration = 1.0 / sample_rate
    self.num_samples = len(mic_sig)
    self.duration = self.num_samples * self.sample_duration
    self.win_len = duration_to_samples(win_len, self.sample_rate) if win_len is not None else self.num_samples
    self.win_len_s = self.win_len / self.sample_rate
    self.win_hop = duration_to_samples(win_hop, self.sample_rate) if win_hop is not None else self.win_len
    self.win_type = win_type if win_type is not None else self.default_win_type
    self.win = self.win_types[self.win_type](self.win_len)
    self.swing_freq = swing_freq
    self.rms_win_len = duration_to_samples(rms_win_len, self.sample_rate)
    
    # self.env_trim = 1 * self.sample_rate
    # self.trim_win_len = 30 * self.sample_rate + 2*self.env_trim
    
    print("normalize mic")
    self.mic_sig = mic_sig
    self.mic_sig = self.mic_sig / np.max(np.abs(self.mic_sig))

    print("normalize render")
    self.render_sig = render_sig
    self.render_sig = self.render_sig / np.max(np.abs(self.render_sig))

    self.results = []
    
    start = 0
    print(f"num_samples = {self.num_samples}")
    while True:
      stop = start + self.win_len
      if stop > self.num_samples:
        break
      
      print(f"start:stop = {start}:{stop}")
      self.results.append(self.analyze_window(start, stop))
      start += self.win_hop

    lags = np.array([r.lag for r in self.results], dtype=np.float32)

    self.lag_sum = np.sum(lags)
    self.count = len(self.results)
    self.lag_mean = self.lag_sum / self.count
    self.lag_stdev = np.std(lags)

    # print(f"lags mean = {self.mean}, stdev = {self.stdev}")

  def analyze_window(self, start, stop, include_signals=True):
    mic_sig = self.mic_sig[start:stop]
    render_sig = self.render_sig[start:stop]
    
    print("compute mic envelope")
    mic_env = envelope_rms(mic_sig, self.rms_win_len)
    
    print("compute render envelope")
    render_env = peak_normalize(envelope_hilbert(render_sig))

    if self.swing_freq is None:
      print("find swing frequency... ", end="")

      n_fft = 2**23
      fax = scipy.fft.rfftfreq(n_fft, 1/self.sample_rate)
      fft = scipy.fft.fft(render_env, n_fft)
      fft_nonneg = fft[:len(fax)]
      # ignore very low frequencies
      fft_nonneg[fax < 0.2] = 0.0

      i_max = np.argmax(np.abs(fft_nonneg))
      swing_freq = fax[i_max]

      print(f"{swing_freq:.03} Hz")

      f_estimate = fax[i_max+1]
    else:
      swing_freq = self.swing_freq
      print(f"use given swing frequency: {swing_freq:.03} Hz")
      f_estimate = swing_freq

    t_estimate = 1/f_estimate
    t_estimate_samp = np.ceil(t_estimate * self.sample_rate)
      
    print("correlate")

    corr, lags = xcorr_unbiased(render_env, mic_env)
    
    corr = corr / np.max(corr)

    corr_raw = np.copy(corr)
    
    corr_win_size_half = round(t_estimate_samp/4)
    # corr[:self.trim_win_len - t_estimate_samp_quarter] = 0.0
    # corr[self.trim_win_len + t_estimate_samp_quarter - 1:] = 0.0
    
    zero_i = len(lags)//2
    corr[:zero_i - corr_win_size_half] = 0.0
    corr[zero_i + corr_win_size_half - 1:] = 0.0
    
    corr_lags = lags
    corr_lags_s = lags / self.sample_rate
    
    i_max_corr = np.argmax(np.abs(corr))

    # lag = (i_max_corr - self.trim_win_len) / self.sample_rate
    lag = corr_lags_s[i_max_corr]
    max_corr = corr[i_max_corr]
    
    print("lag:", lag)

    signals = {}
    if include_signals:
      signals["mic_sig"] = mic_sig
      signals["render_sig"] = render_sig
      signals["mic_env"] = mic_env
      signals["render_env"] = render_env
      signals["corr"] = corr
      signals["corr_raw"] = corr_raw
      signals["corr_lags"] = corr_lags
      signals["corr_lags_s"] = corr_lags_s
    
    return SimpleNamespace(
      start = start,
      stop = stop,
      swing_freq = swing_freq,
      lag = lag,
      max_corr = max_corr,
      **signals
    )
