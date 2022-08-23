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

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--start", type=time_type("--start"), default=0.0)
  arg_parser.add_argument("--length", type=time_type("--length"), default=None)
  arg_parser.add_argument("--window_length", type=int, default=None)
  arg_parser.add_argument("--plot_window", type=int, default=None)
  
  args = arg_parser.parse_args()

  options = SimpleNamespace(
    onsets_hop_length = args.onsets_hop_length,
    analysis_channels = (0, 1),
    analysis_channel_colors = ("#1a85ff", "#d41159"),
    start = args.start,
    length = args.length,
    win_len = args.window_length,
    plot_win = args.plot_window
  )
  
  import tkinter as tk

  root = tk.Tk()
  app = app_swing.App(root, options)
  app.run(args)
  
  root.mainloop()
