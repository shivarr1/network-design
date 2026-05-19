"""
Build monthly demand cell diagrams for Mar, Apr, May 2026.

Produces:
  monthly_demand.png  – three-panel scatter (cells coloured by SLA bucket,
                         size by orders). Dark store marked.
  monthly_demand_map.html – Folium map with one toggleable layer per month.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

df = pd.read_csv("demand_monthly.csv")
print("rows:", len(df), "months:", sorted(df['month'].unique()))

EXISTING_LAT, EXISTING_LNG = 12.904404392534895, 77.6425018662279

# Use a shared scale across months so dot sizes compare apples-to-apples
size_ref = df["orders"].max()
def msize(n):
    return 8 + 700 * (n / size_ref) ** 0.7

months = ["2026-03", "2026-04", "2026-05"]
labels = {"2026-03": "March 2026 (full month)",
          "2026-04": "April 2026 (full month)",
          "2026-05": "May 2026 (1–19)"}

# Use city-wide bbox so all panels share extents
lat_lo, lat_hi = df["lat"].min() - 0.01, df["lat"].max() + 0.01
lng_lo, lng_hi = df["lng"].min() - 0.01, df["lng"].max() + 0.01

# ---------- 1) Three-panel static figure ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
for ax, m in zip(axes, months):
    sub = df[df["month"] == m]
    express = sub[sub["sla_bucket"] == "QuickDelivery"]
    window  = sub[sub["sla_bucket"] == "NonQuickDelivery"]
    ax.scatter(window["lng"], window["lat"],
               s=window["orders"].map(msize), alpha=0.55,
               c="#7a8da3", edgecolors="none",
               label=f"Time-window ({int(window['orders'].sum()):,} orders)")
    ax.scatter(express["lng"], express["lat"],
               s=express["orders"].map(msize), alpha=0.75,
               c="#1f77b4", edgecolors="none",
               label=f"Express ({int(express['orders'].sum()):,} orders)")
    ax.scatter([EXISTING_LNG], [EXISTING_LAT], marker="*",
               s=320, c="#d62728", edgecolors="black", linewidth=0.8,
               label="TR-HSR001 (existing)", zorder=5)
    total = int(sub["orders"].sum())
    cells = len(sub)
    ax.set_title(f"{labels[m]}\n{total:,} orders · {cells} demand cells",
                 fontsize=12)
    ax.set_xlim(lng_lo, lng_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_xlabel("Longitude")
    ax.grid(alpha=0.3)
    ax.set_aspect(1 / np.cos(np.radians(sub["lat"].mean())))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
axes[0].set_ylabel("Latitude")
fig.suptitle("Klydo demand cells — Bangalore, ~1.5 km cells "
             "(dot size ∝ orders, colour = SLA tier)", fontsize=14)
fig.tight_layout()
fig.savefig("monthly_demand.png", dpi=140)
print("wrote monthly_demand.png")

# ---------- 2) Folium map with per-month toggleable layers ----------
center = [df["lat"].mean(), df["lng"].mean()]
fmap = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
folium.Marker(
    [EXISTING_LAT, EXISTING_LNG],
    icon=folium.Icon(color="red", icon="warehouse", prefix="fa"),
    popup="TR-HSR001 (existing dark store)",
    tooltip="TR-HSR001"
).add_to(fmap)

for m in months:
    layer = folium.FeatureGroup(name=labels[m], show=(m == "2026-03"))
    sub = df[df["month"] == m]
    for _, r in sub.iterrows():
        is_express = r["sla_bucket"] == "QuickDelivery"
        folium.CircleMarker(
            location=[r["lat"], r["lng"]],
            radius=2 + 0.45 * (r["orders"] ** 0.6),
            color="#1f77b4" if is_express else "#7a8da3",
            fill=True,
            fill_opacity=0.6,
            weight=0,
            tooltip=f"{m} · {r['sla_bucket']} · {r['orders']:,} orders",
        ).add_to(layer)
    layer.add_to(fmap)

folium.LayerControl(collapsed=False).add_to(fmap)
fmap.save("monthly_demand_map.html")
print("wrote monthly_demand_map.html")

# ---------- 3) Quick monthly summary table ----------
summary = (df.groupby(["month", "sla_bucket"])["orders"].sum().unstack(fill_value=0)
             .assign(total=lambda x: x.sum(axis=1))
             .assign(cells=df.groupby("month")["lat"].count())
             .reset_index())
summary.to_csv("monthly_summary.csv", index=False)
print(summary.to_string(index=False))
