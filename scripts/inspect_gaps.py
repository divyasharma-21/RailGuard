import csv
from datetime import datetime
import collections

path = "data/raw/MetroPT3(AirCompressor).csv"

prev_ts = None
prev_idx = None
gap_dist = collections.Counter()
idx_gaps = collections.Counter()
big_gaps = []  # gaps > 60s
total = 0

with open(path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        total += 1
        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        idx = int(row[""])

        if prev_ts is not None:
            gap = (ts - prev_ts).total_seconds()
            gap_dist[int(gap)] += 1
            if gap > 60:
                big_gaps.append((str(prev_ts), str(ts), gap))
            idx_gap = idx - prev_idx
            idx_gaps[idx_gap] += 1

        prev_ts = ts
        prev_idx = idx

print(f"Total rows: {total}")
print("\n=== TIMESTAMP GAP DISTRIBUTION (seconds, top 15) ===")
for g, c in sorted(gap_dist.most_common(15), key=lambda x: -x[1]):
    print(f"  {g:6d}s  x{c:8d}")

print("\n=== INDEX GAP DISTRIBUTION (top 10) ===")
for g, c in sorted(idx_gaps.most_common(10), key=lambda x: -x[1]):
    print(f"  {g:8d}  x{c:8d}")

print(f"\n=== GAPS > 60s: {len(big_gaps)} ===")
for prev, nxt, g in big_gaps[:20]:
    print(f"  {prev} -> {nxt}  ({g:.0f}s = {g/3600:.2f}h)")
