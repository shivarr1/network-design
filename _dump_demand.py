"""Dump the BQ demand query result (paste-pasted JSON) to CSV."""
import csv, json, sys
from pathlib import Path

# The fields list from BigQuery
SCHEMA = ["lat", "lng", "sla_bucket", "orders_window", "orders_per_day"]
ROWS_RAW = """__ROWS__"""

rows = json.loads(ROWS_RAW)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "demand_real.csv")
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(SCHEMA)
    for r in rows:
        vals = [c["v"] for c in r["f"]]
        w.writerow(vals)
print(f"wrote {len(rows)} rows to {out}")
