"""
Analyze sensor behavior around the 4 documented failure events.
Look at LPS activity, Oil_level, and continuous sensor stats
in the windows before/during/after each failure.
"""
import csv
from datetime import datetime, timedelta
import statistics

FAILURES = [
    ("F1", "2020-04-18 00:00:00", "2020-04-18 23:59:00", "Air leak - High stress", "Maintenance 30Apr 12:00"),
    ("F2", "2020-05-29 23:30:00", "2020-05-30 06:00:00", "Air Leak - High stress", "Maintenance 30Apr 12:00"),
    ("F3", "2020-06-05 10:00:00", "2020-06-07 14:30:00", "Air Leak - High stress", "Maintenance 8Jun 16:00"),
    ("F4", "2020-07-15 14:30:00", "2020-07-15 19:00:00", "Air Leak - High stress", "Maintenance 16Jul 00:00"),
]

FMT = "%Y-%m-%d %H:%M:%S"

def parse(s):
    return datetime.strptime(s, FMT)

# Define analysis windows: 72h before failure start, failure window, 24h after maintenance
windows = []
for fid, start, end, desc, maint in FAILURES:
    fs = parse(start)
    fe = parse(end)
    pre_start = fs - timedelta(hours=72)
    post_end = fe + timedelta(hours=48)
    windows.append((fid, pre_start, fs, fe, post_end, desc))

print("=== FAILURE WINDOW ANALYSIS ===\n")

path = "data/raw/MetroPT3(AirCompressor).csv"

cont_cols = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
dig_cols = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]

# For each failure, collect stats in pre, during, post windows
for fid, pre_start, fs, fe, post_end, desc in windows:
    stats = {"pre": {c: [] for c in cont_cols}, 
             "during": {c: [] for c in cont_cols},
             "post": {c: [] for c in cont_cols}}
    dig_stats = {"pre": {c: 0 for c in dig_cols}, 
                 "during": {c: 0 for c in dig_cols},
                 "post": {c: 0 for c in dig_cols}}
    counts = {"pre": 0, "during": 0, "post": 0}

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = parse(row["timestamp"])
            if ts < pre_start or ts > post_end:
                continue
            if ts < fs:
                period = "pre"
            elif ts <= fe:
                period = "during"
            else:
                period = "post"
            counts[period] += 1
            for c in cont_cols:
                stats[period][c].append(float(row[c]))
            for c in dig_cols:
                dig_stats[period][c] += int(float(row[c]))

    print(f"{'='*60}")
    print(f"Failure {fid}: {desc}")
    print(f"  Pre-failure window:    {pre_start} -> {fs}  ({counts['pre']} rows)")
    print(f"  Failure window:        {fs} -> {fe}  ({counts['during']} rows)")
    print(f"  Post-failure window:   {fe} -> {post_end}  ({counts['post']} rows)")
    print()
    
    print("  CONTINUOUS SENSOR MEANS (pre | during | post):")
    for c in cont_cols:
        pre_m = statistics.mean(stats["pre"][c]) if stats["pre"][c] else float("nan")
        dur_m = statistics.mean(stats["during"][c]) if stats["during"][c] else float("nan")
        post_m = statistics.mean(stats["post"][c]) if stats["post"][c] else float("nan")
        print(f"    {c:20s}  pre={pre_m:8.3f}  during={dur_m:8.3f}  post={post_m:8.3f}")

    print()
    print("  DIGITAL SIGNAL ACTIVATION RATES (% of time active) (pre | during | post):")
    for c in dig_cols:
        def pct(d, period):
            n = counts[period]
            return 100 * d[period][c] / n if n > 0 else 0
        print(f"    {c:20s}  pre={pct(dig_stats,'pre'):5.1f}%  during={pct(dig_stats,'during'):5.1f}%  post={pct(dig_stats,'post'):5.1f}%")
    print()
