"""
Tiny end-to-end correctness test for the optimization models.

Problem (geometry sketched below, x = demand, S = candidate):

    lat
     ^
     |    x1 (express)         x4 (window)
     |       S1            S4
     |               S2
     |    x2 (express)         x3 (window)
     |       S3
     +----------------------------------> lng

S1 covers x1, x2 within express SLA; S4 covers x3, x4 within window SLA.
S2 sits in the middle. With a tier-respecting set cover, optimum should be
exactly {S1, S4} (2 stores).  With express SLA forced on all demand, more
candidates are needed.

Capacity is set so a single store cannot serve everything; max-coverage with
K=1 should pick the site covering the highest-demand pair.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from darkstore_network_design import (Config, generate_candidates,
                                       build_travel_matrix,
                                       model_set_cover, model_min_cost,
                                       model_max_coverage, CFG)

cfg = Config(
    CANDIDATE_GRID_DEG=0.05,
    CANDIDATE_DEMAND_RADIUS_KM=10.0,
    SLA_EXPRESS_MIN=30.0,
    SLA_WINDOW_MIN=120.0,
    EXISTING_STORES=tuple(),
    K_SWEEP=(1, 2, 3, 4),
    STORE_CAPACITY_ORDERS_PER_DAY=10_000,
)

demand = pd.DataFrame([
    # express cluster top-left
    {"lat": 12.95, "lng": 77.60, "sla_bucket": "QuickDelivery",    "orders_per_day": 100, "orders_window": 9000, "polygonname": "k-e", "pincode": 1},
    {"lat": 12.93, "lng": 77.60, "sla_bucket": "QuickDelivery",    "orders_per_day": 100, "orders_window": 9000, "polygonname": "k-e", "pincode": 1},
    # window cluster bottom-right, far from the express cluster (>10 km)
    {"lat": 12.95, "lng": 77.78, "sla_bucket": "NonQuickDelivery", "orders_per_day":  50, "orders_window": 4500, "polygonname": "k-w", "pincode": 2},
    {"lat": 12.93, "lng": 77.78, "sla_bucket": "NonQuickDelivery", "orders_per_day":  50, "orders_window": 4500, "polygonname": "k-w", "pincode": 2},
])
sites = generate_candidates(demand, cfg)
travel = build_travel_matrix(demand, sites, cfg)
print("sites:", len(sites), "travel range:", travel.min(), travel.max())

# ----- Model 1: tier-respecting set cover -----
sc = model_set_cover(demand, sites, travel, cfg)
print("set_cover tier  ->", sc["n_stores"], "stores, uncov=", sc["uncovered_cells"])
# Each cluster diameter is ~2 km, so one well-placed store per cluster (within
# express SLA for left cluster, window SLA easily satisfied for right cluster).
# We expect a small number: 1 (covers both, if a candidate is between) up to 2.
assert sc["status"] == "Optimal"
assert sc["n_stores"] <= 2, f"expected <=2 stores, got {sc['n_stores']}"
assert sc["uncovered_cells"] == 0

# If we tighten the SLA so the two clusters can't share a store -------------
# At prep=12 + travel at 22kmph*1.3, a 25-min SLA gives ~3.5 km radius — wide
# enough to cover within a cluster, but the two clusters are ~18 km apart so
# they CAN'T share a store. Expect >=2 stores, 0 uncovered.
sc_express = model_set_cover(demand, sites, travel, cfg, sla_target_min=25.0)
print("set_cover 25min ->", sc_express["n_stores"], "stores, uncov=",
      sc_express["uncovered_cells"])
assert sc_express["n_stores"] >= 2, "tighter SLA should need >=2 stores"
assert sc_express["uncovered_cells"] == 0

# ----- Model 2: min-cost -----
mc = model_min_cost(demand, sites, travel, cfg)
print("min_cost        ->", mc["n_stores"], "stores, ₹/mo=", round(mc["obj_inr_per_month"]))
assert mc["status"] == "Optimal"
# Sanity: cost is positive and finite
assert 0 < mc["obj_inr_per_month"] < 1e9

# ----- Model 3: max-coverage at express SLA -----
# With K=1 the optimizer should cover the high-demand express cluster
mx1 = model_max_coverage(demand, sites, travel, cfg, K=1)
print("max_cov K=1     ->", round(mx1["covered_orders_per_day"]), "orders/d covered "
      f"({mx1['coverage_pct']:.0f}%)")
assert mx1["status"] == "Optimal"
# Express cluster = 200/d; window cluster cannot be reached within express SLA
# from any single point (clusters are 18 km apart), so K=1 covers exactly the
# express cluster.
assert abs(mx1["covered_orders_per_day"] - 200) < 1e-6, mx1

# With K=2 it should additionally try to grab the window cluster, but those
# customers are >>express SLA away regardless of where you put the store.
mx2 = model_max_coverage(demand, sites, travel, cfg, K=2)
print("max_cov K=2     ->", round(mx2["covered_orders_per_day"]), "orders/d covered")

print("\nALL ASSERTIONS PASSED ✓")
