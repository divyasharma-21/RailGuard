"""
Detailed analysis of LPS (Low Pressure Switch) events - the rare signal (0.3%).
Also analyze Oil_level=0 periods and Caudal_impulses=0 periods as potential failure proxies.
"""
import csv
from datetime import datetime
import collections

path = "data/raw/MetroPT3(AirCompressor).csv"

# Collect LPS=1 intervals, Oil_level=0 intervals, Caudal_impulses=0 intervals
lps_events = []
oil_zero_events = []
caudal_zero_events = []

prev = None
cur_lps = None
cur_oil = None
cur_caudal = None
lps_start = None
oil_start = None
caudal_start = None

with open(path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts = row["timestamp"]
        lps = row["LPS"]
        oil = row["Oil_level"]
        caudal = row["Caudal_impulses"]

        # LPS=1 intervals (anomaly)
        if cur_lps is None:
            cur_lps = lps
            lps_start = ts
        elif lps != cur_lps:
            if cur_lps == "1.0":
                lps_events.append((lps_start, ts))
            cur_lps = lps
            lps_start = ts

        # Oil_level=0 intervals
        if cur_oil is None:
            cur_oil = oil
            oil_start = ts
        elif oil != cur_oil:
            if cur_oil == "0.0":
                oil_zero_events.append((oil_start, ts))
            cur_oil = oil
            oil_start = ts

        # Caudal_impulses=0 intervals
        if cur_caudal is None:
            cur_caudal = caudal
            caudal_start = ts
        elif caudal != cur_caudal:
            if cur_caudal == "0.0":
                caudal_zero_events.append((caudal_start, ts))
            cur_caudal = caudal
            caudal_start = ts

# Compute durations
def dur(a, b):
    fmt = "%Y-%m-%d %H:%M:%S"
    return (datetime.strptime(b, fmt) - datetime.strptime(a, fmt)).total_seconds()

print("=== LPS=1 EVENTS (Low Pressure Alarm?) ===")
print(f"  Count: {len(lps_events)}")
for start, end in lps_events:
    d = dur(start, end)
    print(f"  {start} -> {end}  ({d:.0f}s = {d/60:.1f}min)")

print("\n=== OIL_LEVEL=0 INTERVALS (first 15) ===")
print(f"  Total count: {len(oil_zero_events)}")
durations = [dur(a,b) for a,b in oil_zero_events]
print(f"  Min dur: {min(durations):.0f}s  Max: {max(durations):.0f}s  Mean: {sum(durations)/len(durations):.0f}s")
for start, end in oil_zero_events[:10]:
    d = dur(start, end)
    print(f"  {start} -> {end}  ({d:.0f}s = {d/60:.1f}min)")

print("\n=== CAUDAL_IMPULSES=0 INTERVALS (first 15) ===")
print(f"  Total count: {len(caudal_zero_events)}")
if caudal_zero_events:
    durations2 = [dur(a,b) for a,b in caudal_zero_events]
    print(f"  Min dur: {min(durations2):.0f}s  Max: {max(durations2):.0f}s  Mean: {sum(durations2)/len(durations2):.0f}s")
    for start, end in caudal_zero_events[:10]:
        d = dur(start, end)
        print(f"  {start} -> {end}  ({d:.0f}s = {d/60:.1f}min = {d/3600:.2f}h)")
