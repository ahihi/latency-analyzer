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

from .analysis import SwingAnalysis, truncate_to_even

class FilePlots:
  def __init__(self, ax0, ax1, ax2, analysis, colors=()):
    self.ax0 = ax0
    self.ax1 = ax1
    self.ax2 = ax2
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
    # self.mic_env_plot2 = self.ax0.plot(
    #   self.time, self.analysis.mic_env2,
    #   label=f"mic envelope", color="b", alpha=1, linewidth=0.5
    # )[0]

    self.render_sig_plot = librosa.display.waveshow(
      self.analysis.render_sig, sr=self.analysis.sample_rate,
      label="render", color=self.render_color, alpha=0.25, ax=self.ax1
    )
    self.render_env_plot = self.ax1.plot(
      self.time, self.analysis.render_env,
      label=f"render envelope", color="k", alpha=1, linewidth=0.5
    )[0]

    self.corr_raw_plot = self.ax2.plot(
      self.analysis.corr_lags_s, self.analysis.corr_raw,
      color="k", alpha=0.25, linewidth=0.5
    )
    self.corr_plot = self.ax2.plot(
      self.analysis.corr_lags_s, self.analysis.corr,
      color="k", alpha=1, linewidth=0.5
    )
    # self.corr_max_plot = self.ax2.plot([self.analysis.lag], [self.analysis.max_corr], "o", color="r", markersize=2)
    self.corr_max_vlines = self.ax2.vlines([self.analysis.lag], -1.0, 1.0, color="r", alpha=1, linestyle="-", linewidth=0.5, label=f"max. correlation lag = {self.analysis.lag*1000:.01f} ms")
    
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
    self.gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.0)
    self.env_gs = self.gs[0].subgridspec(2, 1, height_ratios=[1,1], wspace=0.0, hspace=0.0)
    self.ax0 = self.fig.add_subplot(self.env_gs[0])
    self.ax1 = self.fig.add_subplot(self.env_gs[1], sharex=self.ax0)
    self.ax2 = self.fig.add_subplot(self.gs[1])
    self.axs = (self.ax0, self.ax1, self.ax2)
    # self.env_gs.tight_layout(self.fig, pad=4)
    # self.fig.subplots_adjust(hspace=0.0)

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

  def open_file(self, path, force_even=True):
    audio, sample_rate = librosa.load(path, sr=None, mono=False)

    if force_even:
      audio = truncate_to_even(audio)
    
    self.analysis = SwingAnalysis(audio, sample_rate,
      onset_detect_kwargs={
        "units": "time",
        "hop_length": self.options.onsets_hop_length,
        "backtrack": True
      },
      channels=self.options.analysis_channels
    )

    for ax in self.axs:
      ax.clear()

    self.plots = FilePlots(self.ax0, self.ax1, self.ax2, self.analysis, self.options.analysis_channel_colors)

    filename = os.path.basename(path)

    self.ax0.set_title(f"{filename} ({sample_rate} Hz)")

    self.ax0.legend(loc="upper right")
    self.ax0.set_xlabel("time [s]")
    self.ax0.set_ylabel("amplitude")
    self.ax0.get_xaxis().set_visible(False)

    self.ax1.legend(loc="upper right")
    self.ax1.set_xlabel("time [s]")
    self.ax1.set_ylabel("amplitude")

    self.ax2.legend(loc="upper right")
    self.ax2.set_xlabel("lag [s]")
    self.ax2.set_ylabel("correlation")

    
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
