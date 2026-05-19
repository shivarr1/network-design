"""
Run a fast preview of the optimization on the real demand CSV.
Coarser candidate grid + tighter time limits so it finishes in <45s.
"""
import sys, json, time
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
import darkstore_network_design as dn

# Override config for a quick preview
dn.CFG = dn.Config(
    CANDIDATE_GRID_DEG=0.02,         # ~2.2 km grid (smaller MIP)
    CANDIDATE_DEMAND_RADIUS_KM=2.5,
    SOLVER_TIMELIMIT_SEC=25,         # cap each MIP
    K_SWEEP=(1, 2, 3, 5, 8),
)

t0 = time.time()
demand = dn.load_demand(dn.CFG, mock=False, csv="demand_real.csv").reset_index(drop=True)
sites = dn.generate_candidates(demand, dn.CFG).reset_index(drop=True)
travel = dn.build_travel_matrix(demand, sites, dn.CFG)
print(f"sites={len(sites)} demand={len(demand)} travel.shape={travel.shape} "
      f"({time.time()-t0:.1f}s)")

# ---------- Model 1a: tier-respecting set cover ----------
t = time.time()
sc = dn.model_set_cover(demand, sites, travel, dn.CFG)
print(f"\n[set_cover tier-respecting]    stores={sc['n_stores']:>2} "
      f"uncov_cells={sc['uncovered_cells']:>2}  ({time.time()-t:.1f}s)")

# ---------- Model 1b: SLA sweep (express-everywhere) ----------
sweep_results = []
for sla in (30, 45, 60, 90):
    t = time.time()
    r = dn.model_set_cover(demand, sites, travel, dn.CFG, sla_target_min=float(sla))
    sweep_results.append((sla, r["n_stores"], r["uncovered_cells"]))
    print(f"[set_cover @ {sla:>3}min everywhere]  stores={r['n_stores']:>2} "
          f"uncov_cells={r['uncovered_cells']:>2}  ({time.time()-t:.1f}s)")

# ---------- Model 3: max-coverage curve ----------
cov_curve = []
for K in dn.CFG.K_SWEEP:
    t = time.time()
    r = dn.model_max_coverage(demand, sites, travel, dn.CFG, K=K)
    cov_curve.append(r)
    print(f"[max_cov K={K}]  opened={r['n_stores_opened']:>2}  "
          f"covered={r['covered_orders_per_day']:>6.1f}/d "
          f"({r['coverage_pct']:.1f}%)  ({time.time()-t:.1f}s)")

# ---------- Dump opened stores for set cover and biggest max-cov ----------
def _site_rows(idx):
    return sites.iloc[idx][["site_id", "lat", "lng", "fixed"]].to_dict("records")

out = {
    "summary": {
        "total_demand_per_day": float(demand["orders_per_day"].sum()),
        "n_demand_cells": int(len(demand)),
        "n_candidate_sites": int(len(sites)),
    },
    "set_cover_tier_respecting": {
        "n_stores": sc["n_stores"],
        "uncovered_cells": sc["uncovered_cells"],
        "sites": _site_rows(sc["opened"]),
    },
    "set_cover_sla_sweep": [
        {"sla_min": s, "stores_required": n, "uncovered_cells": u}
        for (s, n, u) in sweep_results
    ],
    "max_coverage_curve": [
        {"K": r["K"], "opened": r["n_stores_opened"],
         "covered_per_day": round(r["covered_orders_per_day"], 1),
         "coverage_pct": round(r["coverage_pct"], 1)}
        for r in cov_curve
    ],
    "max_coverage_best_sites": _site_rows(max(cov_curve, key=lambda r: r["K"])["opened"]),
}
with open("preview_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nDONE in {time.time()-t0:.1f}s  ->  preview_results.json")
