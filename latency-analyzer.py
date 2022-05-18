if __name__ == "__main__":  
  import argparse
  from types import SimpleNamespace

  from latency_analyzer.options import Options

  defaults = Options()
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("audio_file", nargs="?")
  arg_parser.add_argument("--onsets_hop_length", type=int, default=defaults.onsets_hop_length)
  
  args = arg_parser.parse_args()

  options = Options()
  options.onsets_hop_length = args.onsets_hop_length
  
  import tkinter as tk
  
  from latency_analyzer.app import App

  root = tk.Tk()
  app = App(root, options)

  if args.audio_file is not None:
    app.open_file(args.audio_file)
  else:
    app.prompt_open_file()
  
  root.mainloop()
