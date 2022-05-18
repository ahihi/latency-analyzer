import argparse
from types import SimpleNamespace

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("audio_file", nargs="?")
arg_parser.add_argument("--onsets_hop_length", type=int, default=512)

args = arg_parser.parse_args()

import librosa
import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sortednp

colors = (
  "#1a85ff",
  "#d41159"
)

# load audio data and analyze

if args.audio_file is not None:
  audio_file = args.audio_file
else:
  audio_file = filedialog.askopenfilename(filetypes=[("WAV files", ".wav")])
  
audio, sample_rate = librosa.load(audio_file, sr=None, mono=False)

n_channels = 1 if len(audio.shape) == 1 else audio.shape[0]

if n_channels < 2:
  raise ValueError(f"expected at least 2 channels, got {n_channels}")

num_samples = audio.shape[1]
sample_duration = 1 / sample_rate
duration = num_samples * sample_duration

time = np.linspace(0.0, duration, num_samples)

def make_channel_data(i):
  onsets = librosa.onset.onset_detect(
    y=audio[i,:], units="samples",
    sr=sample_rate, hop_length=args.onsets_hop_length, backtrack=True
  )
  onset_times = np.vectorize(lambda i: time[i])(onsets)
  return SimpleNamespace(
    index = i,
    audio = audio[i,:],
    onsets = onset_times,
    audio_data = None,
    onsets_data = None
  )

channel_indices = list(range(2))
channels = [make_channel_data(i) for i in channel_indices]
max_abs = np.abs(audio[channel_indices[0]:channel_indices[-1]+1,:]).max()
all_onsets = sortednp.kway_merge(*(channel.onsets for channel in channels))
onset_range = None

# set up window and plot

root = tk.Tk()

fig = matplotlib.figure.Figure((5,4))
ax = fig.add_subplot()

for i, channel in enumerate(channels):
  channel.onsets_data = ax.vlines(channel.onsets, -max_abs, max_abs, label=f"onsets {channel.index}", color=colors[i], alpha=1, linestyle="--")
  channel.audio_data, = ax.plot(time, channel.audio, label=f"channel {channel.index}", color=colors[i], alpha=0.75)

ax.legend()
ax.set_xlabel("time [s]")
ax.set_ylabel("amplitude")

range_rect = patches.Rectangle((0, -max_abs), 0.5, 2*max_abs, linewidth=1, facecolor='#000', alpha=0.125)
ax.add_patch(range_rect)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

# make controls

start_trim_var = tk.IntVar()

def update_vlines(h, x, ymin=None, ymax=None):
  seg_old = h.get_segments()
  if ymin is None:
    ymin = seg_old[0][0, 1]
  if ymax is None:
    ymax = seg_old[0][1, 1]
    
  seg_new = [np.array([[xx, ymin],
                       [xx, ymax]]) for xx in x]
  
  h.set_segments(seg_new)

def start_trim_data(data, start_trim):
  for i, val in enumerate(data):
    if val >= start_trim:
      break
  return data[i:]

def update_start_trim(*args):
  start_trim = start_trim_var.get()
  start_trim_time = start_trim*sample_duration
  for channel in channels:
    update_vlines(channel.onsets_data, start_trim_data(channel.onsets, start_trim_time))
    channel.audio_data.set_data(time[start_trim:], channel.audio[start_trim:])
  ax.set_xlim(time[start_trim], time[-1])
  fig.canvas.draw_idle()

start_trim_ctrl = tk.Scale(
  root,
  from_=0, to=num_samples-2, resolution=1,
  variable=start_trim_var, orient=tk.HORIZONTAL,
  length=400,
  label="start trim [samples]"
)
range_duration_frame = tk.Frame(root)
range_duration_label = tk.Label(range_duration_frame, text="range duration [s]")
range_duration_label.pack(side=tk.TOP, anchor=tk.NW)
range_duration_ctrl = tk.Entry(range_duration_frame,
  # state="disabled",
  # height=1
)
range_duration_ctrl.pack(side=tk.TOP, fill=tk.X)

ctrls = [
  start_trim_ctrl,
  range_duration_frame
]

range_duration_ctrl.bind("<Key>", lambda e: "break")

start_trim_var.trace_add("write", update_start_trim)

def update_range_rect(onset_range):
  if onset_range is not None:
    onset0, onset1 = onset_range
    time0 = all_onsets[onset0] if onset0 is not None else 0.0
    time1 = all_onsets[onset1] if onset1 is not None else duration
    dur = time1-time0
    # print("range duration:", dur)
    range_rect.set(x=time0, width=dur)
    text = str(dur)
  else:
    range_rect.set(x=0.0, width=0.0)
    text = ""

  # range_duration_ctrl.configure(state="normal")
  range_duration_ctrl.delete(0,"end")
  range_duration_ctrl.insert(0, text)
  # range_duration_ctrl.configure(state="disabled")

update_range_rect(onset_range)

def onclick(event):
  t = event.xdata
  if t is None:
    return

  onset0 = None
  for i, onset_time in enumerate(all_onsets):
    if t < onset_time:
      break
    onset0 = i

  onset1 = 0
  if onset0 is not None:
    onset1 = onset0 + 1 if onset0 < len(all_onsets)-1 else None

  onset_range = (onset0, onset1)
  update_range_rect(onset_range)
  
  fig.canvas.draw_idle()
  
fig.canvas.mpl_connect("button_press_event", onclick)

# pack UI elements

for ctrl in ctrls[::-1]:
  ctrl.pack(side=tk.BOTTOM, fill=tk.X)

toolbar.pack(side=tk.BOTTOM, fill=tk.X)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
  
tk.mainloop()
