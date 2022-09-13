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
from tkinter import filedialog, ttk

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
    self.corr_max_vlines = self.ax2.vlines([self.selected_result.lag], -1.0, 1.0, color="r", alpha=1, linestyle="-", linewidth=0.5, label=f"max. correlation lag = {self.selected_result.lag*1000:.02f} ms")
    
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

class EnvsPlot:
  def __init__(self, frame, options, groups, result_info):
    self.frame = frame
    self.options = options
    self.groups = groups

    self.group_key, self.file_i, self.result_i = result_info
    self.analysis = self.groups[self.group_key][self.file_i]
    self.time = None
    self.fig = matplotlib.figure.Figure((16, 7))
    self.gs = gridspec.GridSpec(2, 1, height_ratios=[2,1], wspace=0.0)
    self.env_gs = self.gs[0].subgridspec(2, 1, height_ratios=[1,1], wspace=0.0, hspace=0.0)
    
    self.ax0 = self.fig.add_subplot(self.env_gs[0])
    self.ax1 = self.fig.add_subplot(self.env_gs[1], sharex=self.ax0)
    self.ax2 = self.fig.add_subplot(self.gs[1])
    self.ax2.set_ylim(-1.0, 1.0)

    self.axs = (self.ax0, self.ax1, self.ax2)

    self.shift_down = False
    
    self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
    self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
    self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    self.debug_button_frame = tk.Frame(self.frame)
    self.debug_button = tk.Button(self.debug_button_frame, text="Debug", command=self._on_debug)
    self.debug_button.pack(side=tk.RIGHT)
    self.debug_button_frame.pack(side=tk.TOP, fill=tk.X)

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
    self.canvas.draw() # TODO: refactor?

    # self.menubar = tk.Menu(self.frame)
    # self.menu = SimpleNamespace(
    #   file = tk.Menu(self.menubar, tearoff=0)
    # )
    # self.menu.file.add_command(label="Open", command=lambda *args: self.prompt_open_file(), accelerator="Ctrl+O")
    # self.menu.file.add_separator()
    # self.menu.file.add_command(label="Exit", command=lambda *args: sys.exit(0))
    # self.menubar.add_cascade(label="File", menu=self.menu.file)
    # self.frame.config(menu=self.menubar)
    # self.frame.bind_all("<Control-o>", lambda *args: self.prompt_open_file())
    
    self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
    self.toolbar.update()

    self.start_trim_var = tk.IntVar()
    self.start_trim_var.trace_add("write", lambda *args: self._update_trim())
    self.start_trim_widget = tk.Scale(
      self.frame,
      from_=0, to=0, resolution=1,
      variable=self.start_trim_var, orient=tk.HORIZONTAL,
      label="start trim [samples]"
    )
    self.end_trim_var = tk.IntVar()
    self.end_trim_var.trace_add("write", lambda *args: self._update_trim())
    self.end_trim_widget = tk.Scale(
      self.frame,
      from_=0, to=0, resolution=1,
      variable=self.end_trim_var, orient=tk.HORIZONTAL,
      label="end trim [samples]"
    )

    self.plots = None
    self.plots = FilePlots(self.ax0, self.ax1, self.ax2, self.analysis, self.options.analysis_channel_colors, self.result_i)

    self.ax0.set_title(f"{self.analysis.filename} ({self.analysis.sample_rate} Hz)")
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

    start_frame = int(self.options.start * self.analysis.sample_rate)
    end_frame = start_frame + int(self.options.length * self.analysis.sample_rate) if self.options.length is not None else self.analysis.num_samples-1

    self.start_trim_var.set(start_frame)
    self.end_trim_var.set(end_frame)
    self._update_trim(draw=False)
    
    # pack widgets
    
    widgets = [
      self.start_trim_widget,
      self.end_trim_widget
    ]
    for widget in widgets[::-1]:
      widget.pack(side=tk.BOTTOM, fill=tk.X)

    self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
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

  def _on_debug(self):
    import pdb; pdb.set_trace()

class GroupsBoxPlot:
  def __init__(self, frame, options, groups):
    self.frame = frame
    self.options = options
    self.groups = groups
    self.group_name = self.options.group_name

    self.fig = matplotlib.figure.Figure((16, 7))
    self.ax = self.fig.add_subplot()
    self.x = np.array(sorted(self.groups.keys()))
    
    self.lags_grouped = [np.array([r.lag * 1000 for a in self.groups[k] for r in a.results], dtype=np.float32) for k in self.x]

    self.means_grouped = np.array([np.mean(lags) for lags in self.lags_grouped], dtype=np.float32)
    self.stdevs_grouped = np.array([np.std(lags) for lags in self.lags_grouped], dtype=np.float32)

    self.debug_button_frame = tk.Frame(self.frame)
    self.debug_button = tk.Button(self.debug_button_frame, text="Debug", command=self._on_debug)
    self.debug_button.pack(side=tk.RIGHT)
    self.debug_button_frame.pack(side=tk.TOP, fill=tk.X)

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
    self.canvas.draw() # TODO: refactor?

    self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
    self.toolbar.update()

    self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    width = (np.min(np.ediff1d(self.x)) if len(self.x) > 1 else 1) * 0.75
    self.ax.boxplot(self.lags_grouped, positions=self.x, widths=width)
    
    # self.ax.plot(
    #   self.x, self.means_grouped,
    #   label=f"mean lag", color="k", alpha=1, linewidth=0.5
    # )

    self.ax.set_xlabel(self.group_name)
    # self.ax.set_xticks([i+1 for i, _ in enumerate(self.x)], self.x)
    self.ax.set_ylabel("latency [ms]")

    self.fig.canvas.draw_idle()

  def _on_debug(self):
    import pdb; pdb.set_trace()
      
class BoxPlotWindow:
  def __init__(self, root, options, group_name, group_func):
    self.root = root
    self.root.title(f"analyze-swing")
    self.options = options
    self.group_name = group_name
    self.group_func = group_func
    self.name_filter_re = re.compile(self.options.name_filter_re) if self.options.name_filter_re is not None else None
    self.selectable_results = []
    self.selected_result = None

    self.groups = {}

    self.canvas_frame = tk.Frame(self.root)
    self.plot = None
    
    self.selected_result_frame = tk.Frame(self.root)    
    self.selected_result_frame.pack(side=tk.LEFT, fill=tk.Y) 
    # self.selected_result_frame.pack_propagate(0)
   
    self.selected_result_list = tk.Listbox(
      self.selected_result_frame,
      width=50,
      selectmode=tk.SINGLE
    )
    self.selected_result_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    self.selected_result_list.bind('<<ListboxSelect>>', self._on_selected_result_change)

    self.selected_result_scrollbar = tk.Scrollbar(self.selected_result_frame)
    self.selected_result_scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)

    self.selected_result_list.config(yscrollcommand=self.selected_result_scrollbar.set)
    self.selected_result_scrollbar.config(command=self.selected_result_list.yview)
    
    self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

  def open_path(self, path):
    if os.path.isdir(path):
      files = [os.path.join(path, fn) for fn in os.listdir(path)]
    else:
      files = [path]
    self.open_files(files)
    
  def open_files(self, files):
    groups = {}
    for file_path in files:
      fn = os.path.basename(file_path)
      base, ext = os.path.splitext(fn)
      ext = ext.lower()
      if ext != ".wav":
        continue
      if self.name_filter_re is not None and not self.name_filter_re.search(fn):
        continue
      print(fn)
      try:
        group_key = self.group_func(base)
      except ValueError as e:
        print(e.message)
        continue
      audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
      analysis = SwingAnalysis(
        audio,
        sample_rate,
        self.options.rms_win_len,
        channels=self.options.analysis_channels,
        win_len=self.options.win_len,
        swing_freq=self.options.swing_freq,
        filename=fn
      )
      
      if group_key not in groups:
        groups[group_key] = []
      groups[group_key].append(analysis)

    self.groups = groups
    
    # populate listbox
    
    self.selected_result_list.delete(0, tk.END)
    self.selected_result_list.insert(tk.END, "aggregate")
    self.selectable_results = [None]
    for key in sorted(self.groups.keys()):
      for file_i, analysis in enumerate(self.groups[key]):
        for result_i, result in enumerate(analysis.results):
          self.selected_result_list.insert(tk.END, f"{self.options.group_name} {key}, file {file_i}, window {result_i} -> {result.lag*1000:.02f} ms")
          self.selectable_results.append((key, file_i, result_i))

    self.selected_result_list.activate(0)
    self.selected_result_list.selection_set(0)
    self._on_selected_result_change(None)
    
  def _on_selected_result_change(self, event):
    i = self.selected_result_list.curselection()[0]
    print(f"_on_selected_result_change, i={i}")

    if i == self.selected_result:
      return
    
    self.selected_result = i
    
    for widget in self.canvas_frame.winfo_children():
      widget.destroy()

    result_info = self.selectable_results[i]

    if result_info is None:
      # aggregate
      self.plot = GroupsBoxPlot(self.canvas_frame, self.options, self.groups)
    else:
      self.plot = EnvsPlot(self.canvas_frame, self.options, self.groups, result_info)

def make_re_group_func(group_name, pattern, convert=int):
  regex = re.compile(pattern)
  def _group_func(filename_base):
    m = regex.search(filename_base)
    if not m:
      raise ValueError(f"  no {group_name} found in filename (pattern: {pattern})")
    return convert(m.group(1))
  return _group_func
    
class App:
  def __init__(self, root, options):
    self.root = root
    self.options = options

  def run(self):
    window = BoxPlotWindow(
      self.root, self.options,
      self.options.group_name, make_re_group_func(self.options.group_name, self.options.group_re)
    )
    window.open_path(self.options.audio_file)
