import librosa
import numpy as np
import scipy
import sortednp

def peak_normalize(audio, minimum=-1.0, maximum=1.0):
  return (audio - np.min(audio)) / (np.max(audio) - np.min(audio)) * (maximum - minimum) + minimum

def envelope(audio):
  # analytic = scipy.signal.hilbert(np.concatenate((audio[::-1], audio)))
  # return np.abs(analytic[len(audio):])
  return audio[hi_i]

# https://dsp.stackexchange.com/a/74822
def rolling_rms(x, n):
  x = np.concatenate(([0], x))
  xc = np.cumsum(abs(x)**2);
  return np.sqrt((xc[n:] - xc[:-n]) / n)

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

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

    self.audio = audio
    self.sample_rate = sample_rate
    self.sample_duration = 1.0 / sample_rate
    self.num_samples = audio.shape[1]
    self.duration = self.num_samples * self.sample_duration

    self.channel_indices = channels if channels is not None else range(self.num_channels)

    self.mic_sig = peak_normalize(self.audio[0,:])
    self.render_sig = peak_normalize(self.audio[1,:])

    ixs_lo, ixs_hi = hl_envelopes_idx(self.mic_sig)
    self.mic_env_ixs_hi = ixs_hi
    
    # self.channels = [BonkChannelAnalysis(self.audio[i,:], self.sample_rate, onset_detect_kwargs) for i in self.channel_indices]
    # self.onsets = sortednp.kway_merge(*(channel_analysis.onsets for channel_analysis in self.channels))

    # self.abs_max_amplitude = max(self.channels, key=lambda ca: ca.abs_max_amplitude).abs_max_amplitude
