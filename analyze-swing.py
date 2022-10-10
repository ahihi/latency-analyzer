if __name__ == "__main__":  
  import argparse
  import re
  from types import SimpleNamespace

  from latency_analyzer import app_swing, analysis

  def duration_type(arg_name):
    def _duration_type(arg):
      m = re.match(r"^\d+$", arg)
      if m is not None:
        return SimpleNamespace(samples=int(m.group(0)))
        
      m = re.match(r"^(?:(\d)+)?:(\d+(?:\.\d+)?)$", arg)
      if m is not None:
        return SimpleNamespace(seconds=float(m.group(1) or "0") * 60 + float(m.group(2)))

      raise argparse.ArgumentError(None, f"argument {arg_name}: must be a number of samples, or a duration of the form [minutes]:seconds[.decimals]")

    return _duration_type

  def win_type(arg):
    alternatives = analysis.SwingAnalysis.win_types.keys()
    if arg not in alternatives:
      alternatives_str = ",".join(alternatives)
      raise argparse.ArgumentError(None, f"argument --win_type: must be one of: {alternatives_str}")
    return arg

  def file_channel_type(arg_name):
    def _file_channel_type(arg):
      m = re.match(r"^(?:(\d+):)?(\d+)$", arg)
      if m is None:
        raise argparse.ArgumentError(None, f"argument {arg_name}: must have form [file_index:]channel_index")
      file_i = int(m.group(1) or "0")
      channel_i = int(m.group(2))
      return (file_i, channel_i)
    return _file_channel_type

  def env_method_type(arg_name):
    alternatives = analysis.SwingAnalysis.env_methods.keys()
    def _env_method_type(arg):
      if arg not in alternatives:
        alternatives_str = ",".join(alternatives)
        raise argparse.ArgumentError(None, f"argument {arg_name}: must be one of: {alternatives_str}")
      return arg
    return _env_method_type

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--start", type=duration_type("--start"), default=0.0)
  arg_parser.add_argument("--length", type=duration_type("--length"), default=None)
  arg_parser.add_argument("--mic_channel", type=file_channel_type("--mic_channel"), default=(0, 0))
  arg_parser.add_argument("--mic_env_method", type=env_method_type("--mic_env_method"), default=analysis.SwingAnalysis.default_mic_env_method)
  arg_parser.add_argument("--mic_env_invert",  action=argparse.BooleanOptionalAction)
  arg_parser.add_argument("--render_channel", type=file_channel_type("--render_channel"), default=(0, 1))
  arg_parser.add_argument("--render_env_method", type=env_method_type("--render_env_method"), default=analysis.SwingAnalysis.default_render_env_method)
  arg_parser.add_argument("--render_env_invert",  action=argparse.BooleanOptionalAction)
  arg_parser.add_argument("--win_len", type=duration_type("--win_len"), default=None)
  arg_parser.add_argument("--win_type", type=win_type, default=analysis.SwingAnalysis.default_win_type)
  arg_parser.add_argument("--swing_freq", type=float, default=None)
  arg_parser.add_argument("--rms_win_len", type=duration_type("--rms_win_len"), default=2000)
  arg_parser.add_argument("--bin_re", default=None)
  arg_parser.add_argument("--bin_name", default="bin")
  arg_parser.add_argument("--bin_unit", default=None)
  arg_parser.add_argument("--name_filter_re", default=None)
  arg_parser.add_argument("--font_size", type=float, default=12)
  arg_parser.add_argument("--env_trim", type=duration_type("--env_trim"), default=SimpleNamespace(seconds=0.1))
  arg_parser.add_argument("--allow_negative_lag", action=argparse.BooleanOptionalAction)
  arg_parser.add_argument("--box_plot_means", action=argparse.BooleanOptionalAction)
  arg_parser.add_argument("--ymin", type=float, default=None)
  arg_parser.add_argument("--save_bins_boxplot", default=None)
  arg_parser.add_argument("--save_windows_boxplot", default=None)
  arg_parser.add_argument("--plot_width", type=float, default=16)
  arg_parser.add_argument("--plot_height", type=float, default=7)

  
  args = arg_parser.parse_args()

  args.analysis_channel_colors = ("#1a85ff", "#d41159")
    
  import tkinter as tk

  root = tk.Tk()
  app = app_swing.App(root, args)
  app.run()
  
  root.mainloop()
