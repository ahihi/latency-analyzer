if __name__ == "__main__":  
  import argparse
  import re
  from types import SimpleNamespace

  from latency_analyzer import app_bonk

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--onsets_hop_length", type=int, default=1)
  arg_parser.add_argument("--start", type=time_type("--start"), default=0.0)
  arg_parser.add_argument("--length", type=time_type("--length"), default=None)
  
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

  from latency_analyzer import app_bonk

  root = tk.Tk()
  app = app_bonk.App(root, options)
  app.run(args)
  
  root.mainloop()
