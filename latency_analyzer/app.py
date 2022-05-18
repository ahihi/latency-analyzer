import os
import sys
from types import SimpleNamespace

import librosa
import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

from .analysis import Analysis

class FilePlots:
  def __init__(self, ax, analysis, colors=()):
    self.ax = ax
    self.analysis = analysis
    self.colors = colors
    self.time = np.linspace(0.0, self.analysis.duration, self.analysis.num_samples)
    self.channels = [self._plot_channel(i) for i in range(len(self.analysis.channels))]
    abs_max = self.analysis.abs_max_amplitude
    self.selected_rect = patches.Rectangle((0, -abs_max), 0.0, 2*abs_max, linewidth=0, facecolor="#000", alpha=0.125)
    self.ax.add_patch(self.selected_rect)

  def _plot_channel(self, i):
    abs_max = self.analysis.abs_max_amplitude
    channel_num = self.analysis.channel_indices[i]
    color = self.colors[i] if i < len(self.colors) else "k"
    channel_analysis = self.analysis.channels[i]
    return SimpleNamespace(
      onsets = self.ax.vlines(channel_analysis.onsets, -abs_max, abs_max, label=f"onsets {channel_num}", color=color, alpha=1, linestyle="--"),
      wave = self.ax.plot(self.time, channel_analysis.audio, label=f"wave {channel_num}", color=color, alpha=0.75)[0]
    )

  def update_start_trim(self, trim):
    trim_time = trim*self.analysis.sample_duration
    for channel_analysis, channel_plots in zip(self.analysis.channels, self.channels):
      self._update_vlines(channel_plots.onsets, self._start_trim_data(channel_analysis.onsets, trim_time))
      channel_plots.wave.set_data(self.time[trim:], channel_analysis.audio[trim:])
    self.ax.set_xlim(self.time[trim], self.time[-1])

  def update_selected_onsets(self, start_time, duration):
    self.selected_rect.set(x=start_time, width=duration)
    
  @staticmethod
  def _update_vlines(h, x, ymin=None, ymax=None):
    seg_old = h.get_segments()
    if ymin is None:
      ymin = seg_old[0][0, 1]
    if ymax is None:
      ymax = seg_old[0][1, 1]
    
    seg_new = [np.array([[xx, ymin],
                         [xx, ymax]]) for xx in x]
  
    h.set_segments(seg_new)

  @staticmethod
  def _start_trim_data(data, start_trim):
    for i, val in enumerate(data):
      if val >= start_trim:
        break
    return data[i:]

class App:
  def __init__(self, root, options):
    self.root = root
    self.options = options
    
    self.analysis = None
    self.time = None
    self.fig = matplotlib.figure.Figure((5, 4))
    self.ax = self.fig.add_subplot()

    self.fig.canvas.mpl_connect("button_press_event", self._on_click)
    
    self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
    self.canvas.draw() # TODO: refactor?

    self.menubar = tk.Menu(self.root)
    self.menu = SimpleNamespace(
      file = tk.Menu(self.menubar, tearoff=0)
    )
    self.menu.file.add_command(label="Open", command=lambda *args: self.prompt_open_file(), accelerator="Ctrl+O")
    self.menu.file.add_separator()
    self.menu.file.add_command(label="Exit", command=lambda *args: sys.exit(0))
    self.menubar.add_cascade(label="File", menu=self.menu.file)
    self.root.config(menu=self.menubar)
    self.root.bind_all("<Control-o>", lambda *args: self.prompt_open_file())
    
    self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
    self.toolbar.update()

    self.start_trim_var = tk.IntVar()
    self.start_trim_var.trace_add("write", lambda *args: self._update_start_trim())
    self.start_trim_widget = tk.Scale(
      root,
      from_=0, to=0, resolution=1,
      variable=self.start_trim_var, orient=tk.HORIZONTAL,
      label="start trim [samples]"
    )

    # selected_onsets can be
    #   - None: no selection
    #   - (a, b): onset a to onset b
    #   - (None, b): start of file to onset b
    #   - (a, None): onset a to end of file
    #   - (None, None): start of file to end of file
    self.selected_onsets = None

    self.range_duration_frame = tk.Frame(root)
    self.range_duration_label = tk.Label(self.range_duration_frame, text="selection duration [s]")
    self.range_duration_label.pack(side=tk.TOP, anchor=tk.NW)
    self.range_duration_widget = tk.Entry(self.range_duration_frame)
    self.range_duration_widget.pack(side=tk.TOP, fill=tk.X)
    # self.range_duration_widget.bind("<Key>", lambda e: "break")

    # pack widgets
    
    widgets = [
      self.start_trim_widget,
      self.range_duration_frame
    ]
    for widget in widgets[::-1]:
      widget.pack(side=tk.BOTTOM, fill=tk.X)

    self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

  def open_file(self, path):
    audio, sample_rate = librosa.load(path, sr=None, mono=False)
    self.analysis = Analysis(audio, sample_rate,
      onset_detect_kwargs={
        "units": "time",
        "hop_length": self.options.onsets_hop_length,
        "backtrack": True
      },
      channels=self.options.analysis_channels
    )
    self.ax.clear()
    self.plots = FilePlots(self.ax, self.analysis, self.options.analysis_channel_colors)

    self.ax.legend()
    filename = os.path.basename(path)
    self.ax.set_title(f"{filename} ({sample_rate} Hz)")
    self.ax.set_xlabel("time [s]")
    self.ax.set_ylabel("amplitude")

    self.start_trim_widget.configure(to=max(0, self.analysis.num_samples-2))

    self.start_trim_var.set(0)
    self._update_start_trim(draw=False)
    self.selected_onsets = None
    self._update_selected_onsets(draw=False)

    self.fig.canvas.draw_idle()
    
  def prompt_open_file(self):
    path = filedialog.askopenfilename(filetypes=[("WAV files", ".wav")])
    if path:
      self.open_file(path)

  def _on_click(self, event):
    t = event.xdata
    if t is None or self.analysis is None:
      return

    onset0 = None
    for i, onset_time in enumerate(self.analysis.onsets):
      if t < onset_time:
        break
      onset0 = i

    onset1 = 0
    if onset0 is not None:
      onset1 = onset0 + 1 if onset0 < len(self.analysis.onsets)-1 else None

    self.selected_onsets = (onset0, onset1)
    self._update_selected_onsets()
    
  def _update_start_trim(self, draw=True):
    self.plots.update_start_trim(self.start_trim_var.get())
    if draw:
      self.fig.canvas.draw_idle()

  def _update_selected_onsets(self, draw=True):
    if self.selected_onsets is not None:
      onset0, onset1 = self.selected_onsets
      time0 = self.analysis.onsets[onset0] if onset0 is not None else 0.0
      time1 = self.analysis.onsets[onset1] if onset1 is not None else self.analysis.duration
    else:
      time0 = 0
      time1 = 0

    duration = time1 - time0
    text = str(duration) if duration > 0 else ""

    self.plots.update_selected_onsets(time0, duration)
    self.range_duration_widget.delete(0,"end")
    self.range_duration_widget.insert(0, text)
    if draw:
      self.fig.canvas.draw_idle()
    
