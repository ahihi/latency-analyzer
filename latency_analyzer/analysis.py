import librosa
import numpy as np
import scipy
import sortednp

import time

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
  def __init__(self, audio, sample_rate, onset_detect_kwargs={}, channels=None):
    # audio coming from librosa can have shape (num_channels, num_samples) or (num_samples,)
    is_1d = len(audio.shape) == 1
    self.num_channels = 1 if is_1d else audio.shape[0]

    if self.num_channels < 2:
      raise ValueError(f"expected at least 2 channels, got {self.num_channels}")

    env_trim = 1 * sample_rate
    win_len = 30 * sample_rate + 2*env_trim
    audio = np.take(audio, range(win_len), -1)

    self.audio = audio
    self.sample_rate = sample_rate
    self.sample_duration = 1.0 / sample_rate
    self.num_samples = audio.shape[1]
    self.duration = self.num_samples * self.sample_duration

    self.channel_indices = channels if channels is not None else range(self.num_channels)

    print("peak normalize mic")
    self.mic_sig = self.audio[0,:]
    self.mic_sig = self.mic_sig / np.max(np.abs(self.mic_sig))

    print("peak normalize render")
    self.render_sig = self.audio[1,:]
    self.render_sig = self.render_sig / np.max(np.abs(self.render_sig))
    
    print("compute mic envelope")
    self.mic_env = envelope_rms(self.mic_sig, 2000)
    
    print("compute render envelope")
    self.render_env = peak_normalize(envelope_hilbert(self.render_sig))

    print("find swing frequency... ", end="")

    mic_env_trimmed = np.take(self.mic_env, range(env_trim, win_len-env_trim), -1)
    render_env_trimmed = np.take(self.render_env, range(env_trim, win_len-env_trim), -1)

    n_fft = 2**23
    fax = scipy.fft.rfftfreq(n_fft, 1/self.sample_rate)
    fft = scipy.fft.fft(render_env_trimmed, n_fft)
    fft_nonneg = fft[:len(fax)]
    # ignore very low frequencies
    fft_nonneg[fax < 0.2] = 0.0

    i_max = np.argmax(np.abs(fft_nonneg))
    swing_freq = fax[i_max]
    
    print(f"{swing_freq:.03} Hz")

    f_estimate = fax[i_max+1]
    t_estimate = 1/f_estimate
    t_estimate_samp = np.ceil(t_estimate * self.sample_rate)

    print("correlate")

    corr, lags = xcorr_unbiased(render_env_trimmed, mic_env_trimmed)
    
    corr = corr / np.max(corr)

    self.corr_raw = np.copy(corr)
    
    t_estimate_samp_quarter = round(t_estimate_samp/4)
    # corr[:win_len - t_estimate_samp_quarter] = 0.0
    # corr[win_len + t_estimate_samp_quarter - 1:] = 0.0

    zero_i = len(lags)//2
    corr[:zero_i - round(t_estimate_samp/4)] = 0.0
    corr[zero_i + round(t_estimate_samp/4) - 1:] = 0.0
    
    self.corr = corr
    self.corr_lags = lags
    self.corr_lags_s = lags / self.sample_rate

    i_max_corr = np.argmax(np.abs(corr))

    # lag = (i_max_corr - win_len) / self.sample_rate
    self.lag = self.corr_lags_s[i_max_corr]
    self.max_corr = self.corr[i_max_corr]
    
    print("lag:", self.lag)
    
