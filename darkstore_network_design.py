"""
Dark Store Network Design under SLA Tiers
==========================================

Designs a dark store network for Klydo using order data from BigQuery.
Studies how different SLA promises (express vs time-window) drive different
dark store networks by running three optimization framings side by side:

  Model 1: SET-COVER         – min # stores to give every demand point its
                                tier-appropriate SLA.
  Model 2: MIN-COST CFLP     – cheapest capacitated network respecting SLA.
  Model 3: MAX-COVERAGE      – max orders covered under express SLA given a
                                budget of K stores.

DATA
----
- BigQuery project: klydo-app-b24b7
- Demand: `internal_flows.order_data_flow`  (lat, long, polygonname,
  delivery_promise_minutes, sla_bucket)
- Existing dark store: `hevo_dataset_klydo_app_b24b7_UdHJ.klydo_svc_warehouse`
  (TR-HSR001 @ 12.9044, 77.6425, HSR Layout)

USAGE
-----
    pip install pulp pandas numpy google-cloud-bigquery folium matplotlib
    # Make sure GOOGLE_APPLICATION_CREDENTIALS points at a service account
    # with read access to klydo-app-b24b7.
    python darkstore_network_design.py

    # To run without BigQuery (synthetic data smoke test):
    python darkstore_network_design.py --mock

Outputs land in ./outputs/ next to this script:
  network_summary.csv     – side-by-side comparison of the 3 models
  opened_stores_*.csv     – chosen sites per model
  coverage_curve.png      – marginal coverage vs K for Model 3
  network_map.html        – folium map of all three networks
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CONFIG ---------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- Project / data ---
    BQ_PROJECT: str = "klydo-app-b24b7"
    LOOKBACK_DAYS: int = 90
    DEMAND_GRID_DEG: float = 0.01      # ~1.1 km cells for demand aggregation
    MIN_ORDERS_PER_CELL: int = 3       # drop noise cells (<3 orders in window)

    # --- Existing network ---
    EXISTING_STORES: Tuple[Tuple[str, float, float], ...] = (
        ("TR-HSR001", 12.904404392534895, 77.6425018662279),
    )

    # --- Candidate site generation ---
    CANDIDATE_GRID_DEG: float = 0.015  # ~1.6 km grid of candidate sites
    CANDIDATE_DEMAND_RADIUS_KM: float = 2.5  # candidate must be <=2.5km from
                                             # some demand cell, else dropped

    # --- SLA model ---
    # Travel time = haversine_km * URBAN_DETOUR / AVG_SPEED_KMPH
    URBAN_DETOUR_FACTOR: float = 1.3
    AVG_SPEED_KMPH: float = 22.0       # Bangalore 2W realistic avg
    # Promised SLA includes order prep time inside the store
    PREP_TIME_MIN: float = 12.0
    SLA_EXPRESS_MIN: float = 30.0      # k-express promise
    SLA_WINDOW_MIN: float = 240.0      # time-window promise (4 hours)

    # --- Cost model (INR / month / order, configurable) ---
    FIXED_COST_PER_STORE_MONTH: float = 350_000.0
    VARIABLE_COST_PER_ORDER: float = 35.0
    STORE_CAPACITY_ORDERS_PER_DAY: int = 600
    DAYS_PER_MONTH: int = 30

    # --- Max-coverage sweep ---
    K_SWEEP: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10)

    # --- Solver ---
    SOLVER_TIMELIMIT_SEC: int = 120

CFG = Config()

# ---------------------------------------------------------------------------
# 2. DATA LOADING ---------------------------------------------------------
# ---------------------------------------------------------------------------

def load_demand_bq(cfg: Config) -> pd.DataFrame:
    """Pull aggregated demand from BigQuery."""
    from google.cloud import bigquery

    client = bigquery.Client(project=cfg.BQ_PROJECT)
    grid_step = cfg.DEMAND_GRID_DEG
    sql = f"""
      WITH ord AS (
        SELECT
          combined_order_id,
          ANY_VALUE(lat) AS lat,
          ANY_VALUE(`long`) AS lng,
          ANY_VALUE(polygonname) AS polygonname,
          ANY_VALUE(sla_bucket) AS sla_bucket,
          ANY_VALUE(pincode) AS pincode
        FROM `{cfg.BQ_PROJECT}.internal_flows.order_data_flow`
        WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {cfg.LOOKBACK_DAYS} DAY)
          AND lat IS NOT NULL AND `long` IS NOT NULL
        GROUP BY combined_order_id
      )
      SELECT
        ROUND(lat / {grid_step}) * {grid_step} AS lat_cell,
        ROUND(lng / {grid_step}) * {grid_step} AS lng_cell,
        ANY_VALUE(pincode)    AS pincode,
        ANY_VALUE(polygonname) AS polygonname,
        ANY_VALUE(sla_bucket) AS sla_bucket,
        COUNT(*) AS orders_window,
        COUNT(*) / {cfg.LOOKBACK_DAYS} AS orders_per_day
      FROM ord
      GROUP BY lat_cell, lng_cell
      HAVING orders_window >= {cfg.MIN_ORDERS_PER_CELL}
    """
    df = client.query(sql).to_dataframe()
    df = df.rename(columns={"lat_cell": "lat", "lng_cell": "lng"})
    return df


def load_demand_mock(cfg: Config) -> pd.DataFrame:
    """Synthetic Bangalore demand for smoke testing without BQ access."""
    rng = np.random.default_rng(42)
    # Three demand clusters in Bangalore-like coords
    centers = [(12.91, 77.64, 6, 250),   # HSR / Koramangala (express)
               (12.97, 77.59, 8, 120),   # Majestic / city (window)
               (12.98, 77.72, 7, 100),   # Whitefield (window)
               (13.03, 77.62, 6, 80),    # Hebbal (window)
               (12.90, 77.56, 6, 70)]    # Bannerghatta (window)
    rows = []
    for lat0, lng0, spread_km, n_cells in centers:
        for _ in range(n_cells):
            dlat = rng.normal(0, spread_km / 111.0)
            dlng = rng.normal(0, spread_km / (111.0 * math.cos(math.radians(lat0))))
            lat = round((lat0 + dlat) / cfg.DEMAND_GRID_DEG) * cfg.DEMAND_GRID_DEG
            lng = round((lng0 + dlng) / cfg.DEMAND_GRID_DEG) * cfg.DEMAND_GRID_DEG
            sla = "QuickDelivery" if (lat0, lng0) == (12.91, 77.64) else "NonQuickDelivery"
            rows.append((lat, lng, sla, rng.integers(3, 80)))
    df = pd.DataFrame(rows, columns=["lat", "lng", "sla_bucket", "orders_window"])
    df = (df.groupby(["lat", "lng", "sla_bucket"], as_index=False)
            .agg(orders_window=("orders_window", "sum")))
    df["orders_per_day"] = df["orders_window"] / cfg.LOOKBACK_DAYS
    df["polygonname"] = np.where(df["sla_bucket"] == "QuickDelivery",
                                  "k-express", "k-window")
    df["pincode"] = 560000
    return df


def load_demand_csv(path: str) -> pd.DataFrame:
    """Read pre-aggregated demand from CSV (same shape as load_demand_bq)."""
    df = pd.read_csv(path)
    required = {"lat", "lng", "sla_bucket", "orders_per_day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"demand CSV missing columns: {missing}")
    return df


def load_demand(cfg: Config, mock: bool, csv: str | None = None) -> pd.DataFrame:
    if csv:
        df = load_demand_csv(csv)
    elif mock:
        df = load_demand_mock(cfg)
    else:
        df = load_demand_bq(cfg)
    print(f"[data] demand cells: {len(df):,}  "
          f"orders/day: {df['orders_per_day'].sum():,.0f}  "
          f"express cells: {(df['sla_bucket']=='QuickDelivery').sum():,}  "
          f"window  cells: {(df['sla_bucket']!='QuickDelivery').sum():,}")
    return df

# ---------------------------------------------------------------------------
# 3. GEOMETRY -------------------------------------------------------------
# ---------------------------------------------------------------------------

def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Vectorised great-circle distance in km."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def travel_minutes(km: np.ndarray, cfg: Config) -> np.ndarray:
    return cfg.PREP_TIME_MIN + (km * cfg.URBAN_DETOUR_FACTOR / cfg.AVG_SPEED_KMPH) * 60.0


def generate_candidates(demand: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Uniform lat/lng grid over the demand bbox, plus existing stores forced
    in. Candidates with no nearby demand are dropped to keep MIP tractable."""
    lat_min, lat_max = demand["lat"].min(), demand["lat"].max()
    lng_min, lng_max = demand["lng"].min(), demand["lng"].max()
    step = cfg.CANDIDATE_GRID_DEG
    lats = np.arange(lat_min, lat_max + step, step)
    lngs = np.arange(lng_min, lng_max + step, step)
    grid = pd.DataFrame(
        [(round(la, 4), round(ln, 4)) for la in lats for ln in lngs],
        columns=["lat", "lng"],
    )

    # Filter to candidates within CANDIDATE_DEMAND_RADIUS_KM of any demand cell
    dem_lat = demand["lat"].to_numpy()
    dem_lng = demand["lng"].to_numpy()
    keep = []
    for la, ln in grid.to_numpy():
        d = haversine_km(la, ln, dem_lat, dem_lng)
        if d.min() <= cfg.CANDIDATE_DEMAND_RADIUS_KM:
            keep.append((la, ln))
    grid = pd.DataFrame(keep, columns=["lat", "lng"])
    grid["site_id"] = [f"cand_{i:03d}" for i in range(len(grid))]
    grid["fixed"] = False

    # Force existing stores in as candidates that are always open
    if cfg.EXISTING_STORES:
        existing = pd.DataFrame(
            [(name, la, ln) for name, la, ln in cfg.EXISTING_STORES],
            columns=["site_id", "lat", "lng"],
        )
        existing["fixed"] = True
        sites = pd.concat([existing, grid], ignore_index=True)
    else:
        sites = grid.copy()
    n_fixed = sum(sites["fixed"])
    print(f"[grid] candidate sites: {len(sites):,} "
          f"(fixed: {n_fixed}, generated: {len(grid):,})")
    return sites


def build_travel_matrix(demand: pd.DataFrame,
                        sites: pd.DataFrame,
                        cfg: Config) -> np.ndarray:
    """Returns travel-time matrix M[i,j] in minutes for demand i, site j."""
    dlat = demand["lat"].to_numpy()[:, None]
    dlng = demand["lng"].to_numpy()[:, None]
    slat = sites["lat"].to_numpy()[None, :]
    slng = sites["lng"].to_numpy()[None, :]
    km = haversine_km(dlat, dlng, slat, slng)
    return travel_minutes(km, cfg)

# ---------------------------------------------------------------------------
# 4. OPTIMIZATION MODELS --------------------------------------------------
# ---------------------------------------------------------------------------

def _customer_sla(demand: pd.DataFrame, cfg: Config) -> np.ndarray:
    """SLA budget per demand point in minutes."""
    return np.where(demand["sla_bucket"].to_numpy() == "QuickDelivery",
                    cfg.SLA_EXPRESS_MIN, cfg.SLA_WINDOW_MIN)


def model_set_cover(demand: pd.DataFrame,
                    sites: pd.DataFrame,
                    travel: np.ndarray,
                    cfg: Config,
                    sla_target_min: float | None = None) -> Dict:
    """
    Model 1 – minimum number of dark stores such that every demand cell has at
    least one open store within its SLA.

    If sla_target_min is given, EVERY demand cell must be covered at that SLA
    (the express-everywhere scenario). Otherwise each demand uses its own
    tier (express cells need express SLA, window cells need window SLA).
    """
    import pulp

    n_dem, n_sites = travel.shape
    sla_i = (np.full(n_dem, sla_target_min) if sla_target_min is not None
             else _customer_sla(demand, cfg))
    feasible = travel <= sla_i[:, None]   # demand i can be served by site j

    prob = pulp.LpProblem("min_stores", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_sites)]

    prob += pulp.lpSum(y)
    for j, fixed in enumerate(sites["fixed"]):
        if fixed:
            prob += y[j] == 1
    for i in range(n_dem):
        idx = np.where(feasible[i])[0]
        if len(idx) == 0:
            # demand cell unreachable; skip (e.g. for very tight express SLA
            # the periphery genuinely cannot be covered by any grid candidate)
            continue
        prob += pulp.lpSum(y[j] for j in idx) >= 1

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=cfg.SOLVER_TIMELIMIT_SEC)
    prob.solve(solver)
    opened = [j for j in range(n_sites) if y[j].value() and y[j].value() > 0.5]
    uncovered = sum(1 for i in range(n_dem)
                    if not feasible[i].any() or
                    not any(j in opened for j in np.where(feasible[i])[0]))
    return {
        "status": pulp.LpStatus[prob.status],
        "n_stores": len(opened),
        "opened": opened,
        "uncovered_cells": uncovered,
        "sla_target_min": sla_target_min,
    }


def model_min_cost(demand: pd.DataFrame,
                   sites: pd.DataFrame,
                   travel: np.ndarray,
                   cfg: Config) -> Dict:
    """
    Model 2 – capacitated facility location.
    Each demand point has its own SLA tier (express vs window); we choose
    which sites to open and which demand they serve, minimising
    fixed cost + variable cost.
    """
    import pulp

    n_dem, n_sites = travel.shape
    sla_i = _customer_sla(demand, cfg)
    feasible = travel <= sla_i[:, None]
    demand_pd = demand["orders_per_day"].to_numpy()
    cap = cfg.STORE_CAPACITY_ORDERS_PER_DAY

    prob = pulp.LpProblem("min_cost", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_sites)]
    # x_ij continuous in [0,1] – fraction of demand i served by j
    x = {}
    for i in range(n_dem):
        for j in np.where(feasible[i])[0]:
            x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1)

    # Objective: fixed + variable (variable = per-order cost summed over served)
    monthly_orders = demand_pd * cfg.DAYS_PER_MONTH
    prob += (
        pulp.lpSum(cfg.FIXED_COST_PER_STORE_MONTH * y[j] for j in range(n_sites))
        + pulp.lpSum(cfg.VARIABLE_COST_PER_ORDER * monthly_orders[i] * x[(i, j)]
                     for (i, j) in x)
    )

    for j, fixed in enumerate(sites["fixed"]):
        if fixed:
            prob += y[j] == 1

    # Each demand fully served
    for i in range(n_dem):
        cols = [x[(i, j)] for j in np.where(feasible[i])[0] if (i, j) in x]
        if cols:
            prob += pulp.lpSum(cols) == 1
        # if no feasible site, demand point is unservable; skip

    # Linking
    for (i, j), var in x.items():
        prob += var <= y[j]

    # Capacity
    for j in range(n_sites):
        served = [demand_pd[i] * x[(i, j)] for i in range(n_dem) if (i, j) in x]
        if served:
            prob += pulp.lpSum(served) <= cap * y[j]

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=cfg.SOLVER_TIMELIMIT_SEC)
    prob.solve(solver)

    opened = [j for j in range(n_sites) if y[j].value() and y[j].value() > 0.5]
    assignment = {(i, j): v.value() for (i, j), v in x.items()
                  if v.value() and v.value() > 1e-4}
    return {
        "status": pulp.LpStatus[prob.status],
        "n_stores": len(opened),
        "opened": opened,
        "obj_inr_per_month": pulp.value(prob.objective),
        "assignment": assignment,
    }


def model_max_coverage(demand: pd.DataFrame,
                       sites: pd.DataFrame,
                       travel: np.ndarray,
                       cfg: Config,
                       K: int,
                       sla_min: float | None = None) -> Dict:
    """
    Model 3 – with a budget of K stores, maximise orders covered within
    `sla_min` minutes (defaults to express SLA). Demonstrates the marginal
    value of each extra dark store under a tight SLA.
    """
    import pulp

    n_dem, n_sites = travel.shape
    sla = cfg.SLA_EXPRESS_MIN if sla_min is None else sla_min
    feasible = travel <= sla
    d = demand["orders_per_day"].to_numpy()

    prob = pulp.LpProblem(f"max_cover_K{K}", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_sites)]
    z = [pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(n_dem)]

    prob += pulp.lpSum(d[i] * z[i] for i in range(n_dem))

    # Budget: include fixed-open stores in the count
    prob += pulp.lpSum(y) <= K
    for j, fixed in enumerate(sites["fixed"]):
        if fixed:
            prob += y[j] == 1

    for i in range(n_dem):
        idx = np.where(feasible[i])[0]
        if len(idx) == 0:
            prob += z[i] == 0
        else:
            prob += z[i] <= pulp.lpSum(y[j] for j in idx)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=cfg.SOLVER_TIMELIMIT_SEC)
    prob.solve(solver)
    opened = [j for j in range(n_sites) if y[j].value() and y[j].value() > 0.5]
    covered = sum(d[i] for i in range(n_dem) if z[i].value() and z[i].value() > 0.5)
    return {
        "status": pulp.LpStatus[prob.status],
        "K": K,
        "n_stores_opened": len(opened),
        "opened": opened,
        "covered_orders_per_day": covered,
        "total_demand_per_day": d.sum(),
        "coverage_pct": 100 * covered / d.sum(),
    }

# ---------------------------------------------------------------------------
# 5. REPORTING ------------------------------------------------------------
# ---------------------------------------------------------------------------

def opened_sites_df(sites: pd.DataFrame, indices: List[int]) -> pd.DataFrame:
    return sites.iloc[indices][["site_id", "lat", "lng", "fixed"]].reset_index(drop=True)


def write_map(demand: pd.DataFrame,
              sites: pd.DataFrame,
              networks: Dict[str, List[int]],
              outpath: Path) -> None:
    import folium

    center = [demand["lat"].mean(), demand["lng"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    # Demand as heat dots
    for _, r in demand.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lng"]],
            radius=max(2, min(8, r["orders_per_day"] ** 0.5)),
            color="#666",
            fill=True,
            fill_opacity=0.35,
            weight=0,
            tooltip=f"{r['sla_bucket']} · {r['orders_per_day']:.1f}/d",
        ).add_to(m)

    colors = {"set_cover": "#1f77b4",
              "min_cost":  "#d62728",
              "max_coverage": "#2ca02c"}
    for label, idx in networks.items():
        layer = folium.FeatureGroup(name=label).add_to(m)
        for j in idx:
            s = sites.iloc[j]
            folium.CircleMarker(
                location=[s["lat"], s["lng"]],
                radius=10,
                color=colors.get(label, "black"),
                fill=True,
                fill_opacity=0.85,
                weight=2,
                tooltip=f"{label}: {s['site_id']}",
                popup=f"{label}<br>{s['site_id']}<br>{s['lat']:.4f}, {s['lng']:.4f}",
            ).add_to(layer)
    folium.LayerControl().add_to(m)
    m.save(str(outpath))


def coverage_curve_plot(curve: List[Dict], outpath: Path) -> None:
    import matplotlib.pyplot as plt
    xs = [r["K"] for r in curve]
    ys = [r["coverage_pct"] for r in curve]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", color="#2ca02c")
    ax.set_xlabel("Number of dark stores (K)")
    ax.set_ylabel("Share of daily orders covered under express SLA (%)")
    ax.set_title("Express-SLA coverage vs network size")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)


# ---------------------------------------------------------------------------
# 6. MAIN -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="use synthetic demand instead of BigQuery")
    ap.add_argument("--demand-csv", default=None,
                    help="path to pre-aggregated demand CSV (skips BigQuery)")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Data --------------------------------------------------------------
    demand = load_demand(CFG, mock=args.mock, csv=args.demand_csv).reset_index(drop=True)
    sites = generate_candidates(demand, CFG).reset_index(drop=True)
    travel = build_travel_matrix(demand, sites, CFG)
    print(f"[matrix] travel matrix shape: {travel.shape} "
          f"(min={travel.min():.1f}min, max={travel.max():.1f}min)")

    # 2. Model 1: Set Cover --------------------------------------------------
    print("\n=== Model 1: Min-stores set cover ===")
    # 2a. Tier-respecting cover (express SLA for express cells, window SLA elsewhere)
    sc_tier = model_set_cover(demand, sites, travel, CFG)
    print(f"  tier-respecting        : {sc_tier['n_stores']} stores "
          f"(uncov cells: {sc_tier['uncovered_cells']})")
    # 2b. Express-everywhere scenarios (sensitivity)
    sc_sweep = []
    for sla in (20, 30, 45, 60, 90):
        r = model_set_cover(demand, sites, travel, CFG, sla_target_min=float(sla))
        r["scenario"] = f"all_demand_{sla}min"
        sc_sweep.append(r)
        print(f"  all-demand @ {sla:>3} min   : {r['n_stores']} stores "
              f"(uncov cells: {r['uncovered_cells']})")

    # 3. Model 2: Min-cost CFLP --------------------------------------------
    print("\n=== Model 2: Min-cost capacitated facility location ===")
    mc = model_min_cost(demand, sites, travel, CFG)
    print(f"  stores opened          : {mc['n_stores']}")
    print(f"  monthly cost           : ₹{mc['obj_inr_per_month']:,.0f}")

    # 4. Model 3: Max-coverage sweep ---------------------------------------
    print("\n=== Model 3: Max-coverage at express SLA ===")
    cov_curve = []
    for K in CFG.K_SWEEP:
        r = model_max_coverage(demand, sites, travel, CFG, K=K)
        cov_curve.append(r)
        print(f"  K={K:>2}  opened={r['n_stores_opened']:>2}  "
              f"covered={r['covered_orders_per_day']:>6.0f}/d  "
              f"({r['coverage_pct']:.1f}%)")

    # 5. Persist artefacts -------------------------------------------------
    opened_sites_df(sites, sc_tier["opened"]).to_csv(
        outdir / "opened_stores_set_cover_tier.csv", index=False)
    opened_sites_df(sites, mc["opened"]).to_csv(
        outdir / "opened_stores_min_cost.csv", index=False)
    # use the largest-K max-coverage network as the third illustrative network
    best_cov = max(cov_curve, key=lambda r: r["coverage_pct"])
    opened_sites_df(sites, best_cov["opened"]).to_csv(
        outdir / f"opened_stores_max_cov_K{best_cov['K']}.csv", index=False)

    summary = pd.DataFrame([
        {"model": "set_cover (tier)", "stores": sc_tier["n_stores"],
         "objective": "min stores",
         "cost_per_month": sc_tier["n_stores"] * CFG.FIXED_COST_PER_STORE_MONTH,
         "uncovered_cells": sc_tier["uncovered_cells"],
         "coverage_pct": np.nan},
        {"model": "min_cost CFLP", "stores": mc["n_stores"],
         "objective": "min total cost",
         "cost_per_month": mc["obj_inr_per_month"],
         "uncovered_cells": 0,
         "coverage_pct": 100.0},
        *[
            {"model": f"max_cov K={r['K']}", "stores": r["n_stores_opened"],
             "objective": "max express coverage",
             "cost_per_month": r["n_stores_opened"] * CFG.FIXED_COST_PER_STORE_MONTH,
             "uncovered_cells": np.nan,
             "coverage_pct": r["coverage_pct"]}
            for r in cov_curve
        ],
    ])
    summary.to_csv(outdir / "network_summary.csv", index=False)

    # SLA sensitivity table for set cover
    pd.DataFrame([{
        "scenario": r["scenario"],
        "sla_target_min": r["sla_target_min"],
        "stores_required": r["n_stores"],
        "uncovered_cells": r["uncovered_cells"],
    } for r in sc_sweep]).to_csv(outdir / "sla_sensitivity.csv", index=False)

    # Map + coverage curve
    write_map(demand, sites, {
        "set_cover": sc_tier["opened"],
        "min_cost":  mc["opened"],
        "max_coverage": best_cov["opened"],
    }, outdir / "network_map.html")
    coverage_curve_plot(cov_curve, outdir / "coverage_curve.png")

    print(f"\nArtefacts written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
