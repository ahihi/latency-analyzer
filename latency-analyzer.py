if __name__ == "__main__":  
  import argparse
  import re
  from types import SimpleNamespace

  # fromlist=(None,) lets us import the specified submodule
  # i don't really get it, but if it works...
  app_mod_loaders = {
    "bonk": lambda: __import__("latency_analyzer.app_bonk", fromlist=(None,)),
    "swing": lambda: __import__("latency_analyzer.app_swing", fromlist=(None,))
  }
  def mode_type(arg):
    if arg not in app_mod_loaders:
      modes_str = ",".join(f"'{mode}'" for mode in app_mod_loaders.keys())
      raise argparse.ArgumentError(None, f"argument --mode: must be one of {modes_str}")
    return arg

  def time_type(arg_name):
    def _time_type(arg):
      m = re.match(r"^(?:(\d)+:)?(\d+(?:\.\d+)?)$", arg)
      if m is None:
        raise argparse.ArgumentError(None, f"argument {arg_name}: must be of form [minutes:]seconds[.decimals]")
      return float(m.group(1) or "0") * 60 + float(m.group(2))
    return _time_type
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--onsets_hop_length", type=int, default=1)
  arg_parser.add_argument("--mode", type=mode_type, default="bonk")
  arg_parser.add_argument("--start", type=time_type("--start"), default=0.0)
  arg_parser.add_argument("--length", type=time_type("--length"), default=None)
  
  args = arg_parser.parse_args()

  options = SimpleNamespace(
    onsets_hop_length = args.onsets_hop_length,
    analysis_channels = (0, 1),
    analysis_channel_colors = ("#1a85ff", "#d41159"),
    start = args.start,
    length = args.length
  )
  
  import tkinter as tk

  # from latency_analyzer import app_bonk, app_swing
  app_mod_loader = app_mod_loaders[args.mode]
  app_mod = app_mod_loader()

  root = tk.Tk()
  app = app_mod.App(root, options)

  if args.audio_file is not None:
    app.open_file(args.audio_file)
  else:
    app.prompt_open_file()
  
  root.mainloop()
