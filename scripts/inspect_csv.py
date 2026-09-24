import csv
import statistics

path = "data/raw/MetroPT3(AirCompressor).csv"

cont_cols = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
dig_cols = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]

sampled = {c: [] for c in cont_cols}
dig_unique = {c: set() for c in dig_cols}
timestamps = []
total = 0
index_vals = []

with open(path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        total += 1
        if i % 30 == 0:
            for c in cont_cols:
                sampled[c].append(float(row[c]))
            timestamps.append(row["timestamp"])
            index_vals.append(int(row[""]))
        for c in dig_cols:
            dig_unique[c].add(row[c])

print(f"Total rows: {total}")
print(f"Sampled: {len(timestamps)} rows (1-in-30 systematic)")

# timestamp gap distribution from sampled
from datetime import datetime
ts_parsed = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in timestamps]
gaps = [(ts_parsed[i+1] - ts_parsed[i]).total_seconds() for i in range(len(ts_parsed)-1)]
import collections
gap_dist = collections.Counter(int(g) for g in gaps if g < 60)

print("\n=== TIMESTAMP INFO ===")
print(f"First: {timestamps[0]}")
print(f"Last:  {timestamps[-1]}")
print(f"Index first: {index_vals[0]}  Index last: {index_vals[-1]}")
print("Gap distribution (seconds, sampled):", dict(sorted(gap_dist.most_common(10))))

print("\n=== CONTINUOUS COLUMN STATS (sampled 1-in-30) ===")
for c in cont_cols:
    vals = sampled[c]
    mn, mx, mean = min(vals), max(vals), statistics.mean(vals)
    std = statistics.stdev(vals)
    zeros = sum(1 for v in vals if v == 0.0)
    neg = sum(1 for v in vals if v < 0)
    print(f"  {c:20s}  min={mn:10.4f}  max={mx:10.4f}  mean={mean:10.4f}  std={std:8.4f}  zeros={zeros}  neg={neg}")

print("\n=== DIGITAL COLUMN UNIQUE VALUES ===")
for c in dig_cols:
    print(f"  {c:20s}  {sorted(dig_unique[c])}")
