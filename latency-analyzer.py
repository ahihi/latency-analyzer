if __name__ == "__main__":  
  import argparse
  from types import SimpleNamespace

  from latency_analyzer.options import Options

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
  
  defaults = Options()
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--onsets_hop_length", type=int, default=defaults.onsets_hop_length)
  arg_parser.add_argument("--mode", type=mode_type, default="bonk")
  
  args = arg_parser.parse_args()

  options = Options()
  options.onsets_hop_length = args.onsets_hop_length
  
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
