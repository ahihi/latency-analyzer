def to_ms(ns):
  return ns * 10e-6

def format_ms(ms):
  return f"{ms:.3f} ms"

def format_diff(k):
  return f"{k[0]} -> {k[1]}"

if __name__ == "__main__":  
  import argparse
  import csv

  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("csv_file")

  args = arg_parser.parse_args()

  events = {}
  with open(args.csv_file) as csvfile:  
    reader = csv.DictReader(csvfile, delimiter=",")
    for row in reader:
      event_id = int(row["id"])
      if event_id not in events:
        events[event_id] = {}
      event = events[event_id]
      tag = row["tag"]
      timestamp = int(row["timestamp"])

      if tag in event:
        print(f"ignoring superfluous {k}={v} for event #{event_id}")
        continue

      event[tag] = timestamp

  event_ids = sorted(events.keys())
  diffs_to_compute = [
    ("osc_send", "osc_recv"),
    ("datahandler_enter", "params_done")
  ]
  diffs = {}
  for event_id in event_ids:
    event = events[event_id]
    items = sorted(event.items(), key=lambda item: item[1])
    print(f"#{event_id}:")
    t0 = items[0][1]
    t_last = t0
    for k, t in items:
      dt = t - t_last
      dt0 = t - t0
      dt_ms = to_ms(dt)
      dt0_ms = to_ms(dt0)
      print(f"{k}: {format_ms(dt0_ms)} (+ {format_ms(dt_ms)})")
      t_last = t
    print("----")
    curr_diffs = {(k0,k1): to_ms(event[k1]-event[k0]) for k0, k1 in diffs_to_compute}
    for k in diffs_to_compute:
      v = curr_diffs[k]
      print(f"{format_diff(k)}: {format_ms(v)}")
      if k not in diffs:
        diffs[k] = []
      diffs[k].append(v)
    print()

  print(f"average (n={len(diffs[diffs_to_compute[0]])}):")
  for k in diffs_to_compute:
    k_diffs = diffs[k]
    avg = sum(k_diffs)/len(k_diffs)
    print(f"{format_diff(k)}: {format_ms(avg)}")
  print()
      
