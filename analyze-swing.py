if __name__ == "__main__":  
  import argparse
  import re
  from types import SimpleNamespace

  from latency_analyzer import app_swing

  def time_type(arg_name):
    def _time_type(arg):
      m = re.match(r"^(?:(\d)+:)?(\d+(?:\.\d+)?)$", arg)
      if m is None:
        raise argparse.ArgumentError(None, f"argument {arg_name}: must be of form [minutes:]seconds[.decimals]")
      return float(m.group(1) or "0") * 60 + float(m.group(2))
    return _time_type

  def comparison_type(arg):
    comparisons = {"block_size", "update_rate"}
    if arg not in comparisons:
      alternatives_str = ",".join(comparisons)
      raise argparse.ArgumentError(None, f"argument --comparison: must be one of {alternatives_str}")
    return arg

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--start", type=time_type("--start"), default=0.0)
  arg_parser.add_argument("--length", type=time_type("--length"), default=None)
  arg_parser.add_argument("--window_length", type=int, default=None)
  arg_parser.add_argument("--plot_window", type=int, default=None)
  arg_parser.add_argument("--comparison", type=comparison_type, default=None)
  arg_parser.add_argument("--swing_freq", type=float, default=None)
  arg_parser.add_argument("--plot_groups", action=argparse.BooleanOptionalAction)
  arg_parser.add_argument("--group_re", default=r"(?:^|[ _-])group[ _-]?(\d+)")
  arg_parser.add_argument("--group_name", default="group")
  arg_parser.add_argument("--name_filter_re", default=None)
  
  args = arg_parser.parse_args()

  args.analysis_channels = (0, 1)
  args.analysis_channel_colors = ("#1a85ff", "#d41159")
    
  import tkinter as tk

  root = tk.Tk()
  app = app_swing.App(root, args)
  app.run()
  
  root.mainloop()
