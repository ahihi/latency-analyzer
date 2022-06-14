import argparse
import sys

from pythonosc import dispatcher, osc_server

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="localhost")
  parser.add_argument("--port", default=8765, type=int)
  parser.add_argument("--addr", default="/time")
  parser.add_argument("--output", required=False)
  
  args = parser.parse_args()

  csv_file = None
  if args.output is not None:
    csv_file = open(args.output, "w")
  
  def handle_message(*args):
    # print("received:", args, file=sys.stderr)
    *addr, event_id, tag, timestamp_bytes = args
    timestamp = int.from_bytes(timestamp_bytes, "little", signed=True)
    line = f"{event_id},{tag},{timestamp}"
    print(line)
    if csv_file is not None:
      print(line, file=csv_file)

  try:
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map(args.addr, handle_message)

    server = osc_server.ThreadingOSCUDPServer((args.host, args.port), dispatcher)
    print(f"listening for {args.addr} messages on {args.host}:{args.port}", file=sys.stderr)
    header = "id,tag,timestamp"
    print(header)
    if csv_file is not None:
      print(header, file=csv_file)
    server.serve_forever()
  finally:
    if csv_file is not None:
      csv_file.close()
