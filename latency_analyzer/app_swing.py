import os
import sys
from types import SimpleNamespace

import librosa
import librosa.display
import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

from .analysis import SwingAnalysis

class FilePlots:
  def __init__(self, ax, analysis, colors=()):
    self.ax = ax
    self.analysis = analysis
    self.colors = colors
    self.time = np.linspace(0.0, self.analysis.duration, self.analysis.num_samples)
    
    self.mic_color = self.colors[0] if 0 < len(self.colors) else "k"
    self.render_color = self.colors[1] if 1 < len(self.colors) else "k"

    self.mic_sig_plot = librosa.display.waveshow(
      self.analysis.mic_sig, sr=self.analysis.sample_rate,
      label="mic", color=self.mic_color, alpha=0.75, ax=self.ax
    )
    env_ixs = self.analysis.mic_env_ixs_hi
    self.mic_env_plot = self.ax.plot(
      self.time[env_ixs], self.analysis.mic_sig[env_ixs],
      label=f"mic envelope", color="k", alpha=0.75
    )[0]

    # self.render_plot = librosa.display.waveshow(
    #   self.analysis.render_sig, sr=self.analysis.sample_rate,
    #   label="render", color=self.colors[1], alpha=0.75, ax=self.ax
    # )

  # def _plot_channel(self, i):
  #   channel_num = self.analysis.channel_indices[i]
  #   color = self.colors[i] if i < len(self.colors) else "k"
  #   channel_analysis = self.analysis.channels[i]
  #   return SimpleNamespace(
  #     wave = self.ax.plot(self.time, channel_analysis.audio, label=f"wave {channel_num}", color=color, alpha=0.75)[0]
  #   )

  def update_trim(self, start, end):
    # start_time = start*self.analysis.sample_duration
    # end_time = end*self.analysis.sample_duration
    
    # for channel_analysis, channel_plots in zip(self.analysis.channels, self.channels):
    #   channel_plots.wave.set_data(self.time[start:end+1], channel_analysis.audio[start:end+1])
    # self.ax.set_xlim(self.time[start], self.time[end])
    pass
    
class App:
  def __init__(self, root, options):
    self.root = root
    self.root.title("latency-analyzer (swing)")
    self.options = options
    
    self.analysis = None
    self.time = None
    self.fig = matplotlib.figure.Figure((5, 4))
    self.ax = self.fig.add_subplot()

    self.shift_down = False
    
    self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
    self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
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
    self.start_trim_var.trace_add("write", lambda *args: self._update_trim())
    self.start_trim_widget = tk.Scale(
      root,
      from_=0, to=0, resolution=1,
      variable=self.start_trim_var, orient=tk.HORIZONTAL,
      label="start trim [samples]"
    )
    self.end_trim_var = tk.IntVar()
    self.end_trim_var.trace_add("write", lambda *args: self._update_trim())
    self.end_trim_widget = tk.Scale(
      root,
      from_=0, to=0, resolution=1,
      variable=self.end_trim_var, orient=tk.HORIZONTAL,
      label="end trim [samples]"
    )

    # pack widgets
    
    widgets = [
      self.start_trim_widget,
      self.end_trim_widget
    ]
    for widget in widgets[::-1]:
      widget.pack(side=tk.BOTTOM, fill=tk.X)

    self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

  def open_file(self, path):
    audio, sample_rate = librosa.load(path, sr=None, mono=False)
    self.analysis = SwingAnalysis(audio, sample_rate,
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

    self.start_trim_widget.configure(to=max(0, self.analysis.num_samples-1))
    self.end_trim_widget.configure(to=max(0, self.analysis.num_samples-1))

    self.start_trim_var.set(0)
    self.end_trim_var.set(self.analysis.num_samples-1)
    self._update_trim(draw=False)

    self.fig.canvas.draw_idle()
    
  def prompt_open_file(self):
    path = filedialog.askopenfilename(filetypes=[("WAV files", ".wav")])
    if path:
      self.open_file(path)

  def _on_key_press(self, event):
    if event.key == "shift":
      self.shift_down = True

  def _on_key_release(self, event):
    if event.key == "shift":
      self.shift_down = False
      
  def _on_click(self, event):
    pass
    
  def _update_trim(self, draw=True):
    self.plots.update_trim(self.start_trim_var.get(), self.end_trim_var.get())
    if draw:
      self.fig.canvas.draw_idle()
