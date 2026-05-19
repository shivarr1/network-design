# CLAUDE.md

Guidance for Claude Code sessions working in this repo.

## What this is

Dark-store network design for **Klydo** (q-commerce, Bangalore). Given 90 days of order history, it decides how many dark stores to open, where, and what that costs under different SLA promises. Built around three MILP framings of the same network so the trade-offs are visible side-by-side.

Single operational store today: `TR-HSR001` at `(12.9044, 77.6425)` in HSR Layout. The optimizer always forces it open (`fixed=True`).

## Layout

```
darkstore_network_design.py   # the engine (~614 lines, one file)
_plot_monthly.py              # monthly demand visualisation (Mar/Apr/May 2026)
_smoke_test.py                # end-to-end synthetic-data test
_preview_run.py               # dev helper for fast iteration on a subset
_dump_demand.py               # one-shot demand CSV dumper
demand_real.csv               # cached aggregated demand (skips BQ)
demand_monthly.csv            # monthly breakdown for plots
real_run/                     # outputs from the last full BQ-backed run
mock_outputs/                 # outputs from --mock runs
```

The main script is a pipeline of six top-level sections (numbered in the file): config → data load → geometry → 3 optimization models → reporting → main. No package layout, no tests beyond `_smoke_test.py`.

## Running it

```bash
# Real run (needs GOOGLE_APPLICATION_CREDENTIALS for klydo-app-b24b7)
python darkstore_network_design.py --outdir real_run

# No BigQuery — synthetic Bangalore demand
python darkstore_network_design.py --mock --outdir mock_outputs

# No BigQuery — use cached CSV
python darkstore_network_design.py --demand-csv demand_real.csv --outdir real_run
```

Dependencies: `pulp pandas numpy google-cloud-bigquery folium matplotlib`. CBC solver ships with PuLP.

## Non-obvious things

- **SLA buckets in the data are `QuickDelivery` (=express, 30 min) and `NonQuickDelivery` (=window, 240 min).** The names look generic but they're the canonical values from `internal_flows.order_data_flow`. Don't rename them; downstream filters compare against these exact strings.
- **Travel time = prep + haversine × detour ÷ speed.** Defaults: `PREP_TIME_MIN=12`, `URBAN_DETOUR_FACTOR=1.3`, `AVG_SPEED_KMPH=22`. Bangalore 2W. Tweak in `Config`, not in the formula.
- **Units are mixed on purpose:** capacity is `orders/day` (`STORE_CAPACITY_ORDERS_PER_DAY=600`), cost is `INR/month`. The min-cost model converts via `DAYS_PER_MONTH=30`. If you touch the objective, keep both units in mind.
- **Candidate sites are pre-filtered.** `CANDIDATE_DEMAND_RADIUS_KM=2.5` drops grid points with no nearby demand to keep MIPs tractable. Loosening it explodes the variable count fast.
- **CBC has a 120 s time limit per model** (`SOLVER_TIMELIMIT_SEC`). On full city-scale inputs it usually hits optimal well before that; on stretched grids it may return a feasible-but-suboptimal solution. Check `result["status"]`.
- **The existing store is forced open** in every model via `y[j] == 1` for `sites["fixed"] == True`. If you're testing a greenfield scenario, set `EXISTING_STORES = ()` in `Config`.
- **Demand is grid-aggregated, not per-order.** `DEMAND_GRID_DEG=0.01` ≈ 1.1 km cells. Cells with `<3 orders` in the lookup window are dropped as noise. Order counts inside a cell become `orders_per_day = window_orders / LOOKBACK_DAYS`.
- **Set-cover has two modes.** Default = tier-respecting (express cells need express SLA, window cells need window SLA). Pass `sla_target_min=X` to force every cell to a single SLA (used for the "what if we promised express everywhere?" sweep).
- **Mock demand is deterministic** (`np.random.default_rng(42)`). Five hand-placed clusters; one is express, the rest are window. Useful for code changes, useless for capacity planning.

## When making changes

- Pure local edits + `--mock` is the fast inner loop; full BQ runs take minutes mostly because of CBC.
- New outputs should land in `--outdir`, not in repo root. The current `monthly_*` files in root from `_plot_monthly.py` are an exception — that script writes alongside itself; either follow that pattern or refactor it to take `--outdir` too.
- Don't commit `__pycache__/`, `.DS_Store`, or run outputs that aren't in `real_run/` or `mock_outputs/`. `.gitignore` already covers the first two.

## Data sources

- BigQuery project: `klydo-app-b24b7`
- Demand: `internal_flows.order_data_flow` (cols: `lat`, `long`, `polygonname`, `delivery_promise_minutes`, `sla_bucket`, `pincode`, `order_date`, `combined_order_id`)
- Existing store metadata: `hevo_dataset_klydo_app_b24b7_UdHJ.klydo_svc_warehouse`

## Commit conventions

- Do **not** add `Co-Authored-By: Claude` (or any Claude/Anthropic attribution) trailers to commits. Body only.
