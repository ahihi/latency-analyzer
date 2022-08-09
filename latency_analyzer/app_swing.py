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
  def __init__(self, ax0, ax1, analysis, colors=()):
    self.ax0 = ax0
    self.ax1 = ax1
    self.analysis = analysis
    self.colors = colors
    self.time = np.linspace(0.0, self.analysis.duration, self.analysis.num_samples)
    
    self.mic_color = self.colors[0] if 0 < len(self.colors) else "k"
    self.render_color = self.colors[1] if 1 < len(self.colors) else "k"

    self.mic_sig_plot = librosa.display.waveshow(
      self.analysis.mic_sig, sr=self.analysis.sample_rate,
      label="mic", color=self.mic_color, alpha=0.25, ax=self.ax0
    )
    self.mic_env_plot = self.ax0.plot(
      self.time, self.analysis.mic_env,
      label=f"mic envelope", color="k", alpha=1, linewidth=0.5
    )[0]

    self.render_sig_plot = librosa.display.waveshow(
      self.analysis.render_sig, sr=self.analysis.sample_rate,
      label="render", color=self.render_color, alpha=0.25, ax=self.ax1
    )
    self.render_env_plot = self.ax1.plot(
      self.time, self.analysis.render_env,
      label=f"render envelope", color="k", alpha=1, linewidth=0.5
    )[0]
    
    # self.render_plot = librosa.display.waveshow(
    #   self.analysis.render_sig, sr=self.analysis.sample_rate,
    #   label="render", color=self.colors[1], alpha=0.75, ax=self.ax
    # )

  def update_trim(self, start, end):
    start_time = start*self.analysis.sample_duration
    end_time = end*self.analysis.sample_duration

    self.mic_env_plot.set_data(self.time[start:end+1], self.analysis.mic_env[start:end+1])
    self.render_env_plot.set_data(self.time[start:end+1], self.analysis.render_env[start:end+1])
    
    for ax in (self.ax0, self.ax1):
      ax.set_xlim(self.time[start], self.time[end])
    
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
    self.fig = matplotlib.figure.Figure((16, 7))
    self.gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    self.ax0 = self.fig.add_subplot(self.gs[0])
    self.ax1 = self.fig.add_subplot(self.gs[1])
    self.fig.tight_layout(pad=4)
    # self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.0)

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
    self.ax0.clear()
    self.ax1.clear()
    self.plots = FilePlots(self.ax0, self.ax1, self.analysis, self.options.analysis_channel_colors)

    filename = os.path.basename(path)

    self.ax0.legend()
    self.ax0.set_title(f"{filename} ({sample_rate} Hz)")
    self.ax0.set_xlabel("time [s]")
    self.ax0.set_ylabel("amplitude")

    self.ax1.legend()
    self.ax1.set_xlabel("time [s]")
    self.ax1.set_ylabel("amplitude")

    self.start_trim_widget.configure(to=max(0, self.analysis.num_samples-1))
    self.end_trim_widget.configure(to=max(0, self.analysis.num_samples-1))

    start_frame = int(self.options.start * sample_rate)
    end_frame = start_frame + int(self.options.length * sample_rate) if self.options.length is not None else self.analysis.num_samples-1
    
    self.start_trim_var.set(start_frame)
    self.end_trim_var.set(end_frame)
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
