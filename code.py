import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")


# CONFIG

TAU = 0.5  # Cosine similarity threshold
K = 3  # City count threshold
YEARS = [2023, 2024, 2025]


# CATEGORY ASSIGNMENT & COLORS

CATEGORY_RULES = {
    "Fruits & Vegetables": [
        "banana",
        "apple",
        "orange",
        "mango",
        "tomato",
        "potato",
        "onion",
        "garlic",
        "ginger",
    ],
    "Meat & Protein": ["beef", "mutton", "chicken", "meat", "egg"],
    "Dairy & Fats": ["milk", "yogurt", "curd", "butter", "cheese", "ghee"],
    "Grains & Staples": [
        "rice",
        "wheat",
        "flour",
        "bread",
        "sugar",
        "salt",
        "lentil",
        "daal",
        "pulse",
    ],
    "Beverages": ["tea", "coffee", "beverage", "lipton", "drink"],
    "Cooking Essentials": ["oil", "spice", "chili", "turmeric", "masala"],
}
CATEGORY_COLORS = {
    "Fruits & Vegetables": "#2ecc71",  # green
    "Meat & Protein": "#e74c3c",  # red
    "Dairy & Fats": "#f1c40f",  # yellow
    "Grains & Staples": "#e67e22",  # orange
    "Beverages": "#9b59b6",  # purple
    "Cooking Essentials": "#1abc9c",  # teal
    "Other": "#3498db",  # blue
}


def assign_category(item):
    item_l = item.lower()
    for cat, keywords in CATEGORY_RULES.items():
        if any(k in item_l for k in keywords):
            return cat
    return "Other"


def node_colors(G, df):
    cat_map = dict(zip(df["item"], df["category"]))
    return [CATEGORY_COLORS.get(cat_map.get(n, "Other"), "#3498db") for n in G.nodes()]


# STEP 1: LOAD DATA

print("Loading data...")
df = pd.read_csv("data.csv")
df["category"] = df["item"].apply(assign_category)
print(
    f"Records: {len(df)} | Items: {df['item'].nunique()} | Cities: {df['city'].nunique()}"
)


# STEP 2: PRICE CHANGE VECTORS


def compute_price_changes(df, year):
    yd = df[df["year"] == year].sort_values("month")
    vectors = {}
    for (item, city), grp in yd.groupby(["item", "city"]):
        if len(grp) < 2:
            continue
        changes = np.diff(grp.sort_values("month")["price"].values)
        if len(changes) > 0:
            vectors[(item, city)] = changes
    return vectors


print("\nComputing price change vectors...")
price_vectors = {y: compute_price_changes(df, y) for y in YEARS}
for y in YEARS:
    print(f"  {y}: {len(price_vectors[y])} item-city pairs")


# STEP 3: SIMILARITY MATRIX


def compute_similarity(vectors, tau):
    city_sims = {}
    cos_avgs = {}
    cities = set(c for _, c in vectors.keys())
    for city in cities:
        items = [i for (i, c) in vectors.keys() if c == city]
        if len(items) < 2:
            continue
        vecs = [vectors[(i, city)] for i in items]
        max_len = max(len(v) for v in vecs)
        vecs = np.array([np.pad(v, (0, max_len - len(v))) for v in vecs])
        sim = cosine_similarity(vecs)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s = sim[i, j]
                if s >= tau:
                    pair = tuple(sorted([items[i], items[j]]))
                    city_sims.setdefault(pair, []).append(city)
                    cos_avgs.setdefault(pair, []).append(s)
    avg_cos = {p: np.mean(v) for p, v in cos_avgs.items()}
    return city_sims, avg_cos


print("\nComputing similarity matrices...")
city_sims = {}
avg_cosines = {}
for y in YEARS:
    city_sims[y], avg_cosines[y] = compute_similarity(price_vectors[y], TAU)
    print(f"  {y}: {len(city_sims[y])} similar pairs")


# STEP 4: BUILD GRAPHS


def build_graph(sims, avg_cos=None, k=K, weight="count"):
    G = nx.Graph()
    for pair, cities in sims.items():
        if len(cities) >= k:
            if weight == "count":
                G.add_edge(*pair, weight=len(cities))
            elif weight == "cosine" and avg_cos:
                G.add_edge(*pair, weight=avg_cos.get(pair, 0))
            else:
                G.add_edge(*pair)
    return G


print("\nBuilding graphs...")
graphs = {y: build_graph(city_sims[y], avg_cosines[y], K, "count") for y in YEARS}
graphs_cosine = {
    y: build_graph(city_sims[y], avg_cosines[y], K, "cosine") for y in YEARS
}
for y in YEARS:
    print(
        f"  {y}: {graphs[y].number_of_nodes()} nodes, {graphs[y].number_of_edges()} edges"
    )


# STEP 5: CENTRALITIES


def compute_centralities(G):
    if G.number_of_nodes() == 0:
        return {}, {}, {}
    return (
        nx.degree_centrality(G),
        nx.closeness_centrality(G),
        nx.betweenness_centrality(G),
    )


print("\nComputing centralities...")
centralities = {}
for y in YEARS:
    d, c, b = compute_centralities(graphs[y])
    centralities[y] = {"degree": d, "closeness": c, "betweenness": b}


# STEP 6: VISUALIZATIONS (9 FIGURES)

plt.style.use("seaborn-v0_8-whitegrid")


def draw_network(
    G, pos, sizes, colors, title, fname, edge_widths=None, cmap=None, show_legend=False
):
    plt.figure(figsize=(16, 13))
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.4,
        width=edge_widths if edge_widths is not None else 1.0,
        edge_color="#555555",
    )
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=colors,
        cmap=cmap,
        alpha=0.9,
        edgecolors="black",
        linewidths=1.2,
    )
    # Label ALL nodes
    labels = {n: n[:30] for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, font_size=8, font_color="black", font_weight="bold"
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    if show_legend:
        handles = [
            plt.scatter(
                [], [], s=120, c=col, label=cat, edgecolors="black", linewidths=1
            )
            for cat, col in CATEGORY_COLORS.items()
        ]
        plt.legend(
            handles=handles,
            loc="upper left",
            fontsize=9,
            frameon=True,
            title="Category",
        )
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {fname}")


print("\nCreating visualizations...")
year = 2024
G = graphs[year]
G_cos = graphs_cosine[year]
deg = centralities[year]["degree"]
clo = centralities[year]["closeness"]
bet = centralities[year]["betweenness"]
pos = nx.spring_layout(G, k=2, iterations=80, seed=42)
sizes = [deg.get(n, 0) * 4000 + 200 for n in G.nodes()]
colors = node_colors(G, df)

# Figure 1: Edge width = city count
edge_w = [G[u][v]["weight"] * 0.4 for u, v in G.edges()]
draw_network(
    G,
    pos,
    sizes,
    colors,
    f"Figure 1: Weighted Network (Edge Width ∝ # Cities Co-Moving) — {year}, τ={TAU}, K={K}",
    "figure1_weighted_city_count.png",
    edge_widths=edge_w,
    show_legend=True,
)

# Figure 2: Edge width = avg cosine similarity
pos_cos = nx.spring_layout(G_cos, k=2, iterations=80, seed=42)
sizes_cos = [deg.get(n, 0) * 4000 + 200 for n in G_cos.nodes()]
colors_cos = node_colors(G_cos, df)
edge_w_cos = [G_cos[u][v]["weight"] * 6 for u, v in G_cos.edges()]
draw_network(
    G_cos,
    pos_cos,
    sizes_cos,
    colors_cos,
    f"Figure 2: Weighted Network (Edge Width ∝ Avg Cosine Similarity) — {year}, τ={TAU}, K={K}",
    "figure2_weighted_avg_cosine.png",
    edge_widths=edge_w_cos,
    show_legend=True,
)

# Figure 3: Centrality network — color=closeness, size=degree, labels=all
clo_colors = [clo.get(n, 0) for n in G.nodes()]
plt.figure(figsize=(16, 13))
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8, edge_color="#555555")
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_size=sizes,
    node_color=clo_colors,
    cmap="viridis",
    alpha=0.9,
    edgecolors="black",
    linewidths=1.2,
)
labels = {n: n[:30] for n in G.nodes()}
nx.draw_networkx_labels(
    G, pos, labels, font_size=8, font_color="black", font_weight="bold"
)
plt.colorbar(nodes, label="Closeness Centrality", shrink=0.6)
plt.title(
    f"Figure 3: Centrality Network — Color: Closeness, Size: Degree — {year}",
    fontsize=14,
    fontweight="bold",
)
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "figure3_centrality_network.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved figure3_centrality_network.png")

# Figure 4: Complement network
G_comp = nx.complement(G)
G_comp.remove_edges_from(nx.selfloop_edges(G_comp))
if G_comp.number_of_edges() > 0:
    pos_comp = nx.spring_layout(G_comp, k=1.5, iterations=80, seed=42)
    sizes_comp = [300] * G_comp.number_of_nodes()
    colors_comp = node_colors(G_comp, df)
    draw_network(
        G_comp,
        pos_comp,
        sizes_comp,
        colors_comp,
        f"Figure 4: Complement Network — Edges represent ABSENCE of co-movement — {year}",
        "figure4_complement_network.png",
        show_legend=True,
    )

# Figure 5: Connected components (each component a different color)
components = list(nx.connected_components(G))
comp_palette = plt.cm.Set3(np.linspace(0, 1, max(len(components), 1)))
node_to_comp_color = {}
for idx, comp in enumerate(components):
    for n in comp:
        node_to_comp_color[n] = comp_palette[idx]
comp_colors = [node_to_comp_color[n] for n in G.nodes()]
plt.figure(figsize=(16, 13))
nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8, edge_color="#555555")
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=sizes,
    node_color=comp_colors,
    alpha=0.9,
    edgecolors="black",
    linewidths=1.2,
)
nx.draw_networkx_labels(
    G,
    pos,
    {n: n[:30] for n in G.nodes()},
    font_size=8,
    font_color="black",
    font_weight="bold",
)
plt.title(
    f"Figure 5: Connected Components — {len(components)} components — {year}, τ={TAU}, K={K}",
    fontsize=14,
    fontweight="bold",
)
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "figure5_connected_components.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved figure5_connected_components.png")

# Figure 6: Cosine similarity heatmap (top 20 by degree)
top20 = [i for i, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:20]]
mat = np.zeros((len(top20), len(top20)))
for i, a in enumerate(top20):
    for j, b in enumerate(top20):
        if i == j:
            mat[i, j] = 1.0
        else:
            pair = tuple(sorted([a, b]))
            mat[i, j] = avg_cosines[year].get(pair, 0)
plt.figure(figsize=(13, 11))
sns.heatmap(
    mat,
    annot=True,
    fmt=".2f",
    cmap="plasma",
    xticklabels=[i[:18] for i in top20],
    yticklabels=[i[:18] for i in top20],
    cbar_kws={"label": "Cosine Similarity"},
    linewidths=0.3,
    linecolor="white",
)
plt.title(
    f"Figure 6: Cosine Similarity Heatmap — Top 20 Items by Degree — {year}, τ={TAU}",
    fontsize=14,
    fontweight="bold",
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "figure6_similarity_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved figure6_similarity_heatmap.png")

# Figure 7: Temporal degree centrality (top 15 by avg degree across years)
all_items = set().union(*[centralities[y]["degree"].keys() for y in YEARS])
avg_deg = {
    it: np.mean([centralities[y]["degree"].get(it, 0) for y in YEARS])
    for it in all_items
}
top15 = [i for i, _ in sorted(avg_deg.items(), key=lambda x: x[1], reverse=True)[:15]]
fig, ax = plt.subplots(figsize=(15, 8))
x = np.arange(len(top15))
width = 0.27
year_colors = ["#3498db", "#e67e22", "#2ecc71"]
for idx, y in enumerate(YEARS):
    vals = [centralities[y]["degree"].get(it, 0) for it in top15]
    ax.bar(
        x + idx * width,
        vals,
        width,
        label=str(y),
        color=year_colors[idx],
        edgecolor="black",
        linewidth=0.6,
    )
ax.set_xlabel("Items", fontsize=12)
ax.set_ylabel("Degree Centrality", fontsize=12)
ax.set_title(
    "Figure 7: Temporal Analysis — Degree Centrality Across 2023–2025 (Top 15 Items)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xticks(x + width)
ax.set_xticklabels([it[:22] for it in top15], rotation=45, ha="right", fontsize=9)
ax.legend(title="Year")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    "figure7_temporal_degree.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved figure7_temporal_degree.png")

# Figure 8: Sensitivity to tau (0.1 -> 0.95, K=3)
tau_vals = np.linspace(0.1, 0.95, 18)
res_tau = []
for t in tau_vals:
    for y in YEARS:
        sims, _ = compute_similarity(price_vectors[y], t)
        Gx = build_graph(sims, None, 3, "none")
        res_tau.append({"tau": t, "year": y, "edges": Gx.number_of_edges()})
df_tau = pd.DataFrame(res_tau)
fig, ax = plt.subplots(figsize=(12, 7))
for idx, y in enumerate(YEARS):
    d = df_tau[df_tau["year"] == y]
    ax.plot(
        d["tau"],
        d["edges"],
        marker="o",
        label=str(y),
        color=year_colors[idx],
        linewidth=2.2,
        markersize=7,
    )
ax.axvline(x=0.5, color="red", linestyle="--", linewidth=2, alpha=0.7, label="τ = 0.5")
ax.set_xlabel("Similarity Threshold (τ)", fontsize=12)
ax.set_ylabel("Number of Edges", fontsize=12)
ax.set_title(
    "Figure 8: Sensitivity Analysis — Edge Count vs. τ (K=3)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(title="Year")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "figure8_sensitivity_tau.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved figure8_sensitivity_tau.png")

# Figure 9: Sensitivity to K (1 -> 17, tau=0.5)
res_K = []
for k in range(1, 18):
    for y in YEARS:
        Gx = build_graph(city_sims[y], avg_cosines[y], k, "count")
        res_K.append({"K": k, "year": y, "edges": Gx.number_of_edges()})
df_K = pd.DataFrame(res_K)
fig, ax = plt.subplots(figsize=(12, 7))
for idx, y in enumerate(YEARS):
    d = df_K[df_K["year"] == y]
    ax.plot(
        d["K"],
        d["edges"],
        marker="s",
        label=str(y),
        color=year_colors[idx],
        linewidth=2.2,
        markersize=7,
    )
ax.axvline(x=3, color="red", linestyle="--", linewidth=2, alpha=0.7, label="K = 3")
ax.set_xlabel("City Count Threshold (K)", fontsize=12)
ax.set_ylabel("Number of Edges", fontsize=12)
ax.set_title(
    "Figure 9: Threshold Comparison — Edge Count vs. K (τ=0.5)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(title="Year")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "figure9_sensitivity_K.png", dpi=200, bbox_inches="tight", facecolor="white"
)
plt.close()
print("  Saved figure9_sensitivity_K.png")


# STEP 7: EXPORT CSVS

print("\nExporting CSVs...")
for y in YEARS:
    rows = []
    for ct in ["degree", "closeness", "betweenness"]:
        for item, score in sorted(
            centralities[y][ct].items(), key=lambda x: x[1], reverse=True
        ):
            rows.append(
                {"year": y, "item": item, "centrality_type": ct, "score": score}
            )
    pd.DataFrame(rows).to_csv(f"centrality_rankings_{y}.csv", index=False)
    print(f"  Saved centrality_rankings_{y}.csv")

for y in YEARS:
    rows = [
        {"item1": u, "item2": v, "weight": d.get("weight", 1)}
        for u, v, d in graphs[y].edges(data=True)
    ]
    pd.DataFrame(rows).to_csv(f"edges_{y}.csv", index=False)
    print(f"  Saved edges_{y}.csv")

summary = [
    {
        "year": y,
        "nodes": graphs[y].number_of_nodes(),
        "edges": graphs[y].number_of_edges(),
        "density": nx.density(graphs[y]),
        "avg_degree": sum(dict(graphs[y].degree()).values())
        / max(graphs[y].number_of_nodes(), 1),
    }
    for y in YEARS
]
pd.DataFrame(summary).to_csv("network_summary.csv", index=False)
print("  Saved network_summary.csv")

print("\n" + "=" * 60)
print("DONE — generated 9 figures and 7 CSV files.")
print("=" * 60)
