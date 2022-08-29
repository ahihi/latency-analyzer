import os
import re
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
  def __init__(self, ax0, ax1, ax2, analysis, colors=(), plot_win=0):
    self.ax0 = ax0
    self.ax1 = ax1
    self.ax2 = ax2
    self.analysis = analysis
    self.selected_result = self.analysis.results[plot_win]
    self.colors = colors
    self.time = np.linspace(0.0, self.analysis.win_len_s, self.analysis.win_len)
    
    self.mic_color = self.colors[0] if 0 < len(self.colors) else "k"
    self.render_color = self.colors[1] if 1 < len(self.colors) else "k"

    self.mic_sig_plot = librosa.display.waveshow(
      self.selected_result.mic_sig, sr=self.analysis.sample_rate,
      label="mic", color=self.mic_color, alpha=0.25, ax=self.ax0
    )
    self.mic_env_plot = self.ax0.plot(
      self.time, self.selected_result.mic_env,
      label=f"mic envelope", color="k", alpha=1, linewidth=0.5
    )[0]

    self.render_sig_plot = librosa.display.waveshow(
      self.selected_result.render_sig, sr=self.analysis.sample_rate,
      label="render", color=self.render_color, alpha=0.25, ax=self.ax1
    )
    self.render_env_plot = self.ax1.plot(
      self.time, self.selected_result.render_env,
      label=f"render envelope", color="k", alpha=1, linewidth=0.5
    )[0]

    self.corr_raw_plot = self.ax2.plot(
      self.selected_result.corr_lags_s, self.selected_result.corr_raw,
      color="k", alpha=0.25, linewidth=0.5
    )
    self.corr_plot = self.ax2.plot(
      self.selected_result.corr_lags_s, self.selected_result.corr,
      color="k", alpha=1, linewidth=0.5
    )
    # self.corr_max_plot = self.ax2.plot([self.selected_result.lag], [self.selected_result.max_corr], "o", color="r", markersize=2)
    self.corr_max_vlines = self.ax2.vlines([self.selected_result.lag], -1.0, 1.0, color="r", alpha=1, linestyle="-", linewidth=0.5, label=f"max. correlation lag = {self.selected_result.lag*1000:.01f} ms")
    
  def update_trim(self, start, end):
    start_time = start*self.analysis.sample_duration
    end_time = end*self.analysis.sample_duration

    self.mic_env_plot.set_data(self.time[start:end+1], self.selected_result.mic_env[start:end+1])
    self.render_env_plot.set_data(self.time[start:end+1], self.selected_result.render_env[start:end+1])
    
    for ax in (self.ax0, self.ax1):
      ax.set_xlim(self.time[start], self.time[end])
    
    # for channel_analysis, channel_plots in zip(self.analysis.channels, self.channels):
    #   channel_plots.wave.set_data(self.time[start:end+1], channel_analysis.audio[start:end+1])
    # self.ax.set_xlim(self.time[start], self.time[end])
    pass

class EnvsPlotWindow:
  def __init__(self, root, options):
    self.root = root
    self.root.title("analyze-swing: envelopes & correlation")
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
    
    self.analysis = SwingAnalysis(
      audio,
      sample_rate,
      channels=self.options.analysis_channels,
      win_len=self.options.window_length
    )

    for ax in self.axs:
      ax.clear()

    self.plots = None
    self.plots = FilePlots(self.ax0, self.ax1, self.ax2, self.analysis, self.options.analysis_channel_colors, self.options.plot_window)

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

    self.start_trim_widget.configure(to=max(0, self.analysis.win_len-1))
    self.end_trim_widget.configure(to=max(0, self.analysis.win_len-1))

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

class BoxPlotWindow:
  def __init__(self, root, options, group_name, group_func):
    self.root = root
    self.root.title(f"analyze-swing: {group_name}")
    self.options = options
    self.group_name = group_name
    self.group_func = group_func

    self.groups = {}
    self.x = None
    self.fig = matplotlib.figure.Figure((16, 7))
    self.ax = self.fig.add_subplot()
    self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
    self.canvas.draw() # TODO: refactor?

    self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
    self.toolbar.update()

    self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
  def open_dir(self, path):
    groups = {}
    for fn in os.listdir(path):
      base, ext = os.path.splitext(fn)
      ext = ext.lower()
      if ext != ".wav":
        continue
      print(fn)
      try:
        group_key = self.group_func(base)
      except ValueError as e:
        print(e.message)
        continue
      file_path = os.path.join(path, fn)
      audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
      analysis = SwingAnalysis(
        audio,
        sample_rate,
        channels=self.options.analysis_channels,
        win_len=self.options.window_length
      )
      
      if group_key not in groups:
        groups[group_key] = []
      groups[group_key].append(analysis)

    self.groups = groups
    
    self.x = np.array(sorted(groups.keys()))

    self.lags_grouped = [np.array([r.lag for a in self.groups[k] for r in a.results], dtype=np.float32) for k in self.x]

    self.means_grouped = np.array([np.mean(lags) for lags in self.lags_grouped], dtype=np.float32)
    self.stdevs_grouped = np.array([np.std(lags) for lags in self.lags_grouped], dtype=np.float32)

    width = np.min(np.ediff1d(self.x)) * 0.75
    self.ax.boxplot(self.lags_grouped, positions=self.x, widths=width)
    
    # self.ax.plot(
    #   self.x, self.means_grouped,
    #   label=f"mean lag", color="k", alpha=1, linewidth=0.5
    # )

    self.ax.set_xlabel("block size")
    # self.ax.set_xticks([i+1 for i, _ in enumerate(self.x)], self.x)
    self.ax.set_ylabel("latency")
    
    self.fig.canvas.draw_idle()

def make_re_group_func(group_name, pattern, convert=int):
  regex = re.compile(pattern)
  def _group_func(filename_base):
    m = regex.match(filename_base)
    if not m:
      raise ValueError(f"  no {group_name} found in filename (pattern: {pattern})")
    return convert(m.group(1))
  return _group_func
    
class App:
  def __init__(self, root, options):
    self.root = root
    self.options = options

  def run(self):
    if self.options.plot_block_size:
      group_name = "block size"
      window = BoxPlotWindow(
        self.root, self.options,
        group_name, make_re_group_func(group_name, self.options.block_size_re)
      )
      window.open_dir(self.options.audio_file)
    elif self.options.plot_window is not None:
      window = EnvsPlotWindow(self.root, self.options)
      if self.options.audio_file is not None:
        window.open_file(self.options.audio_file)
      else:
        window.prompt_open_file()
    else:
      pass

