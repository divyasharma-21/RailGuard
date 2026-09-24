"""
Inspect class balance of digital signals and check for known failure annotations.
Also inspect the last 50 rows to see data tail.
"""
import csv
from datetime import datetime
import collections

path = "data/raw/MetroPT3(AirCompressor).csv"

dig_cols = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
dig_counts = {c: collections.Counter() for c in dig_cols}
total = 0
rows_tail = []

with open(path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        total += 1
        for c in dig_cols:
            dig_counts[c][row[c]] += 1
        # keep last 50 for tail inspection
        rows_tail.append(row)
        if len(rows_tail) > 50:
            rows_tail.pop(0)

print(f"Total rows: {total}")
print("\n=== DIGITAL SIGNAL CLASS COUNTS ===")
for c in dig_cols:
    cnts = dig_counts[c]
    v0 = int(cnts.get("0.0", 0))
    v1 = int(cnts.get("1.0", 0))
    pct1 = 100 * v1 / total
    print(f"  {c:20s}  0={v0:8d} ({100-pct1:5.1f}%)   1={v1:8d} ({pct1:5.1f}%)")

print("\n=== LAST 5 ROWS ===")
for row in rows_tail[-5:]:
    print(dict(row))

# Look at Oil_level=0 events more carefully - count transitions
print("\n=== TRANSITION COUNTS (0->1 or 1->0) ===")
prev_vals = None
transitions = {c: 0 for c in dig_cols}
trans_times = {c: [] for c in dig_cols}

with open(path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        cur = {c: row[c] for c in dig_cols}
        ts = row["timestamp"]
        if prev_vals is not None:
            for c in dig_cols:
                if cur[c] != prev_vals[c]:
                    transitions[c] += 1
                    trans_times[c].append((prev_vals[c], cur[c], ts))
        prev_vals = cur

for c in dig_cols:
    print(f"  {c:20s}  transitions={transitions[c]}")
    # print first few transitions
    for t in trans_times[c][:3]:
        print(f"    {t[0]}->{t[1]} at {t[2]}")
