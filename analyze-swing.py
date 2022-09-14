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

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--start", type=duration_type("--start"), default=0.0)
  arg_parser.add_argument("--length", type=duration_type("--length"), default=None)
  arg_parser.add_argument("--win_len", type=duration_type("--win_len"), default=None)
  arg_parser.add_argument("--win_type", type=win_type, default=analysis.SwingAnalysis.default_win_type)
  arg_parser.add_argument("--swing_freq", type=float, default=None)
  arg_parser.add_argument("--rms_win_len", type=duration_type("--rms_win_len"), default=2000)
  arg_parser.add_argument("--bin_re", default=r"(?:^|[ _-])bin[ _-]?(\d+)")
  arg_parser.add_argument("--bin_name", default="bin")
  arg_parser.add_argument("--bin_unit", default=None)
  arg_parser.add_argument("--name_filter_re", default=None)
  arg_parser.add_argument("--font_size", type=int, default=12)
  
  args = arg_parser.parse_args()

  args.analysis_channels = (0, 1)
  args.analysis_channel_colors = ("#1a85ff", "#d41159")
    
  import tkinter as tk

  root = tk.Tk()
  app = app_swing.App(root, args)
  app.run()
  
  root.mainloop()
