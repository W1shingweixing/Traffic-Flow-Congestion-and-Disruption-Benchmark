"""
=============================================================================
DATA5925 Group 14 — Member 3: Spatial Graph & Centrality Analysis
=============================================================================
Datasets:
    road_df   → roaddetail_2025.csv
    align_df  → station_incident_alignment_2025.csv
    haz_df    → hazards_2025_flattened.csv
    flow_df   → hourly_flow_2025.csv
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports & data loading
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import json, os

road_df  = pd.read_csv("roaddetail_2025.csv")
road_df["station_id"] = road_df["station_id"].astype(str).str.strip()
road_df  = road_df.dropna(subset=["wgs84_latitude", "wgs84_longitude"])

align_df = pd.read_csv("station_incident_alignment_2025.csv")
align_df["station_id"] = align_df["station_id"].astype(str).str.strip()
align_df["dt"] = pd.to_datetime(align_df["dt"], dayfirst=True, errors="coerce")

haz_df   = pd.read_csv("hazards_2025_flattened.csv")
haz_df["dt"] = pd.to_datetime(haz_df["dt"], errors="coerce")

flow_df  = pd.read_csv("hourly_flow_2025.csv")
flow_df["timestamp"]  = pd.to_datetime(flow_df["timestamp"])
flow_df["station_id"] = flow_df["station_id"].astype(str).str.strip()

print(f"road_df  : {len(road_df):>6} sensor stations")
print(f"align_df : {len(align_df):>6} station-incident pairs")
print(f"haz_df   : {len(haz_df):>6} hazard records  ({haz_df['hazard_type'].nunique()} types)")
print(f"flow_df  : {len(flow_df):>6} flow records    ({flow_df['station_id'].nunique()} stations)")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR = "#0d1117"

def apply_dark_theme(ax):
    """Apply consistent dark background + white axis styling."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres (Haversine formula)."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def safe_corr(a, b):
    """Pearson r guarded against constant columns (returns nan instead of error)."""
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Build road network graph
# ─────────────────────────────────────────────────────────────────────────────
# Strategy A — same-road stations within ROAD_DIST km
# Strategy B — k-NN spatial nearest neighbours within KNN_MAX km

ROAD_DIST = 50.0   # km
KNN_K     = 3
KNN_MAX   = 15.0   # km

ids   = road_df["station_id"].values
roads = road_df["road_name"].values
lats  = road_df["wgs84_latitude"].values
lons  = road_df["wgs84_longitude"].values
n     = len(ids)

# Pre-compute full pairwise distance matrix
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = D[j, i] = haversine_km(lats[i], lons[i], lats[j], lons[j])

G = nx.DiGraph()
for _, row in road_df.iterrows():
    G.add_node(
        row["station_id"],
        lat=row["wgs84_latitude"],
        lon=row["wgs84_longitude"],
        road=str(row["road_name"]),
        suburb=str(row["suburb"]),
        device=str(row["device_type"]),
        quality=int(row["quality_rating"]),
    )

# Strategy A: same-road directed edges
same_road_pairs = 0
for i in range(n):
    for j in range(i + 1, n):
        if roads[i] == roads[j] and D[i, j] <= ROAD_DIST:
            G.add_edge(ids[i], ids[j], weight=D[i, j], etype="same_road")
            G.add_edge(ids[j], ids[i], weight=D[i, j], etype="same_road")
            same_road_pairs += 1

# Strategy B: k-NN spatial edges
knn_pairs = 0
for i in range(n):
    sorted_js = np.argsort(D[i])
    count = 0
    for j in sorted_js:
        if j == i:
            continue
        if D[i, j] > KNN_MAX or count >= KNN_K:
            break
        if not G.has_edge(ids[i], ids[j]):
            G.add_edge(ids[i], ids[j], weight=D[i, j], etype="spatial_knn")
            G.add_edge(ids[j], ids[i], weight=D[i, j], etype="spatial_knn")
            knn_pairs += 1
        count += 1

UG         = G.to_undirected()
components = list(nx.connected_components(UG))
giant      = max(components, key=len)
G_giant    = G.subgraph(giant).copy()
UG_giant   = G_giant.to_undirected()

print(f"\nGraph built:")
print(f"  Nodes      : {G.number_of_nodes()}")
print(f"  Edges      : {G.number_of_edges()} "
      f"(same-road {same_road_pairs*2} + spatial-kNN {knn_pairs*2})")
print(f"  Components : {len(components)}, giant = {len(giant)} nodes")

# Save adjacency matrix and node order
node_order = list(G.nodes())
adj_matrix = nx.to_numpy_array(G, nodelist=node_order, weight="weight")
np.save("node_order_nsw.npy", np.array(node_order))
np.save("adj_matrix_nsw.npy", adj_matrix)
print(f"\nSaved: node_order_nsw.npy ({len(node_order)} nodes)")
print(f"Saved: adj_matrix_nsw.npy {adj_matrix.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Centrality metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing centrality metrics …")
deg_cent = nx.degree_centrality(UG_giant)
bet_cent = nx.betweenness_centrality(UG_giant, normalized=True, weight="weight")
clo_cent = nx.closeness_centrality(UG_giant, distance="weight")

cent_df = pd.DataFrame({
    "station_id"  : list(deg_cent.keys()),
    "degree_cent" : list(deg_cent.values()),
    "between_cent": [bet_cent.get(nd, 0) for nd in deg_cent],
    "close_cent"  : [clo_cent.get(nd, 0) for nd in deg_cent],
})

# Min-max normalise each metric to [0, 1]
for col in ["degree_cent", "between_cent", "close_cent"]:
    mn, mx = cent_df[col].min(), cent_df[col].max()
    cent_df[col + "_norm"] = (cent_df[col] - mn) / (mx - mn + 1e-9)

# Composite score = equal-weight average of the three normalised metrics
cent_df["composite_score"] = cent_df[
    ["degree_cent_norm", "between_cent_norm", "close_cent_norm"]
].mean(axis=1)

cent_df = cent_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
cent_df["rank"] = cent_df.index + 1

# Merge road metadata
cent_df = cent_df.merge(
    road_df[["station_id", "road_name", "suburb",
             "wgs84_latitude", "wgs84_longitude"]],
    on="station_id", how="left",
)

# Merge incident counts
inc_cnt = (align_df
    .groupby("station_id")
    .size()
    .reset_index(name="incident_count"))
cent_df = cent_df.merge(inc_cnt, on="station_id", how="left")
cent_df["incident_count"] = cent_df["incident_count"].fillna(0).astype(int)

print("\nTop-10 stations by composite centrality:")
show = ["rank", "station_id", "road_name", "suburb",
        "degree_cent", "between_cent", "close_cent",
        "composite_score", "incident_count"]
print(cent_df[show].head(10).to_string(index=False))

# Pearson correlation: centrality vs. historical incident counts
r_between = safe_corr(cent_df["between_cent"], cent_df["incident_count"])
r_degree  = safe_corr(cent_df["degree_cent"],  cent_df["incident_count"])
r_close   = safe_corr(cent_df["close_cent"],   cent_df["incident_count"])
print(f"\nValidation (centrality vs incident count):")
print(f"  Betweenness r = {r_between:.3f}")
print(f"  Degree      r = {r_degree:.3f}")
print(f"  Closeness   r = {r_close:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Figure 1 — Road network map
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 1: road network map …")

node_list = list(G.nodes())
lons_all  = [G.nodes[nd]["lon"] for nd in node_list]
lats_all  = [G.nodes[nd]["lat"] for nd in node_list]

score_map = dict(zip(cent_df["station_id"], cent_df["composite_score"]))
scores    = np.array([score_map.get(nd, 0.0) for nd in node_list])
norm      = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())

fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Edges: coloured by edge type
for u, v, d in G.edges(data=True):
    col = "#2a3a6a" if d.get("etype") == "same_road" else "#1a3a1a"
    ax.plot(
        [G.nodes[u]["lon"], G.nodes[v]["lon"]],
        [G.nodes[u]["lat"], G.nodes[v]["lat"]],
        color=col, lw=0.7, alpha=0.55, zorder=1,
    )

# Nodes: size and colour encode composite centrality
sc = ax.scatter(
    lons_all, lats_all,
    c=scores, cmap=cm.plasma, norm=norm,
    s=80 + 300 * norm(scores),
    edgecolors="white", linewidths=0.4, zorder=3,
)

# Historical incident overlay (sample of 400)
haz_geo = haz_df.dropna(subset=["latitude", "longitude"])
haz_s   = haz_geo.sample(min(400, len(haz_geo)), random_state=42)
ax.scatter(
    haz_s["longitude"], haz_s["latitude"],
    c="red", marker="+", s=14, alpha=0.30, zorder=2,
    label="Historical incident (sample)",
)

# Annotate top-5 stations
for _, row in cent_df.head(5).iterrows():
    if pd.notna(row.get("wgs84_longitude")):
        ax.annotate(
            f"#{int(row['rank'])} {row['station_id']}\n"
            f"{str(row['road_name'])[:22]}",
            xy=(row["wgs84_longitude"], row["wgs84_latitude"]),
            xytext=(10, 8), textcoords="offset points",
            color="white", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="#12122e", ec="#8888ff", lw=0.9),
            zorder=6,
        )

cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("Composite centrality score", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

p_road = mpatches.Patch(color="#2a3a6a", label="Same-road edge")
p_knn  = mpatches.Patch(color="#1a3a1a", label="Spatial k-NN edge")
p_haz  = mpatches.Patch(color="red",     label="Historical incident (sample)")
ax.legend(handles=[p_road, p_knn, p_haz],
          loc="lower right", facecolor="#1a1a2e",
          labelcolor="white", fontsize=9)

apply_dark_theme(ax)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude",  fontsize=10)
ax.set_title(
    "NSW road sensor network\n"
    "Node size/colour = composite centrality  ·  "
    "Blue edges = same road  ·  Green edges = spatial k-NN  ·  Red + = incident",
    fontsize=11, pad=12,
)

plt.tight_layout()
plt.savefig("fig1_road_network.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.show()
print("  → fig1_road_network.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Figure 2 — Centrality comparison: Top-15 stations, grouped bar
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 2: centrality comparison bar chart …")

top15  = cent_df.head(15).copy()
labels = [f"#{int(r)}\n{sid}"
          for r, sid in zip(top15["rank"], top15["station_id"])]
x, w   = np.arange(len(labels)), 0.26

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(BG_COLOR)

ax.bar(x - w, top15["degree_cent_norm"],  w,
       label="Degree centrality",      color="#4c8eff", alpha=0.85)
ax.bar(x,     top15["between_cent_norm"], w,
       label="Betweenness centrality", color="#ff6b6b", alpha=0.85)
ax.bar(x + w, top15["close_cent_norm"],   w,
       label="Closeness centrality",   color="#51cf66", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(labels, color="white", fontsize=8)
ax.set_ylabel("Normalised centrality score")
ax.set_title("Top-15 sensor stations — three centrality metrics (normalised)")
apply_dark_theme(ax)
ax.legend(facecolor="#1a1a2e", labelcolor="white")

plt.tight_layout()
plt.savefig("fig2_centrality_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.show()
print("  → fig2_centrality_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure 3 — Centrality vs. historical incident count (validation)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3: centrality vs incident count scatter …")

pairs = [
    ("between_cent", "Betweenness centrality", f"r = {r_between:.3f}"),
    ("degree_cent",  "Degree centrality",      f"r = {r_degree:.3f}"),
    ("close_cent",   "Closeness centrality",   f"r = {r_close:.3f}"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG_COLOR)
rng = np.random.RandomState(42)

for ax, (col, label, r_str) in zip(axes, pairs):
    jitter = rng.uniform(-0.3, 0.3, len(cent_df))
    ax.scatter(
        cent_df[col],
        cent_df["incident_count"] + jitter,
        c=cent_df["composite_score"], cmap="plasma",
        alpha=0.75, s=50, edgecolors="none",
    )
    x_arr = cent_df[col].values
    y_arr = cent_df["incident_count"].values
    if x_arr.std() > 1e-9:
        m = np.cov(x_arr, y_arr)[0, 1] / (np.var(x_arr) + 1e-9)
        b = y_arr.mean() - m * x_arr.mean()
        xs = np.linspace(x_arr.min(), x_arr.max(), 60)
        ax.plot(xs, m * xs + b, "w--", lw=1.2, label="Regression")

    apply_dark_theme(ax)
    ax.set_xlabel(label)
    ax.set_ylabel("Historical incident count" if ax is axes[0] else "")
    ax.set_title(f"{label}\n{r_str}")

fig.suptitle("Centrality metrics vs. historical incident count — method validation",
             color="white", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig3_centrality_vs_incidents.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.show()
print("  → fig3_centrality_vs_incidents.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Figure 4 — Historical hazard distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 4: historical incident distribution …")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_COLOR)

# Top-10 hazard types
top_types  = haz_df["hazard_type"].value_counts().head(10)
bar_colors = cm.plasma(np.linspace(0.2, 0.9, len(top_types)))
ax1.barh(top_types.index[::-1], top_types.values[::-1],
         color=bar_colors[::-1], alpha=0.85)
ax1.set_xlabel("Number of records")
ax1.set_title(f"Top-10 hazard types\n({len(haz_df):,} total records)")
apply_dark_theme(ax1)

# Hourly distribution
haz_valid = haz_df.dropna(subset=["dt"])
hour_cnt  = haz_valid["dt"].dt.hour.value_counts().sort_index()
ax2.bar(
    hour_cnt.index, hour_cnt.values,
    color=cm.plasma(hour_cnt.values / hour_cnt.values.max()),
    alpha=0.85,
)
ax2.set_xlabel("Hour of day (0–23)")
ax2.set_ylabel("Incident count")
ax2.set_title("Incidents by hour\n(Twin peaks 8–9 AM & 16–17 PM)")
apply_dark_theme(ax2)

plt.tight_layout()
plt.savefig("fig4_incident_distribution.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.show()
print("  → fig4_incident_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Export result tables
# ─────────────────────────────────────────────────────────────────────────────
cent_df.to_csv("centrality_rankings.csv", index=False)
print("\nExported: centrality_rankings.csv")

summary = {
    "graph_nodes"              : G.number_of_nodes(),
    "graph_edges"              : G.number_of_edges(),
    "giant_component_nodes"    : len(giant),
    "n_connected_components"   : len(components),
    "dist_threshold_same_road_km": ROAD_DIST,
    "knn_k"                    : KNN_K,
    "knn_max_km"               : KNN_MAX,
    "corr_betweenness_incidents": round(r_between, 4) if not np.isnan(r_between) else None,
    "corr_degree_incidents"    : round(r_degree, 4)   if not np.isnan(r_degree)  else None,
    "corr_closeness_incidents" : round(r_close, 4)    if not np.isnan(r_close)   else None,
}
with open("graph_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Exported: graph_summary.json")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Member 3 — Road Network Graph & Centrality Analysis")
print("=" * 65)
print(f"\nNetwork  : {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
print(f"           Giant component = {len(giant)} nodes")
print(f"\nTop-5 most critical stations (composite centrality):")
for _, row in cent_df.head(5).iterrows():
    print(f"  #{int(row['rank'])}  {row['station_id']:15s}  "
          f"{str(row['road_name'])[:35]:35s}  "
          f"score={row['composite_score']:.4f}  "
          f"incidents={int(row['incident_count'])}")
print(f"\nFigures saved:")
for i, desc in enumerate([
    "Road network map (node size = centrality, incident overlay)",
    "Top-15 centrality comparison (3-metric grouped bar)",
    "Centrality vs. incident count (validation scatter)",
    "Historical incident types & hourly distribution",
], 1):
    print(f"  fig{i}_*.png  — {desc}")
print(f"\nFiles saved:")
print(f"  node_order_nsw.npy      — {len(node_order)} station IDs in fixed order")
print(f"  adj_matrix_nsw.npy      — {adj_matrix.shape} adjacency matrix")
print(f"  centrality_rankings.csv — centrality scores for all stations")
print(f"  graph_summary.json      — graph statistics")
