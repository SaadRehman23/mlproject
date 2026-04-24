# ============================================================
# ML STAGE 2 — Customer Segmentation Using Clustering
# Student: Mohammed Saad  |  ID: F20230214
# Dataset: Mall Customers Dataset
# ============================================================


# ── 1. Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot style
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD & INSPECT DATA
# ─────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: LOAD & INSPECT DATA")
print("=" * 60)

# ── Download dataset (Kaggle CSV) ─────────────────────────────────────────
# If running locally, place Mall_Customers.csv in the same folder.
# In Google Colab you can upload via: from google.colab import files; files.upload()
# Or use the Kaggle API:
#   !pip install kaggle
#   !kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python
#   !unzip customer-segmentation-tutorial-in-python.zip

CSV_PATH = "Mall_Customers.csv"   # <-- update path if needed

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    # Fallback: generate synthetic data matching the real dataset's statistics
    print("[INFO] Mall_Customers.csv not found — generating synthetic data for demo.\n"
          "       Download the real CSV from Kaggle and set CSV_PATH correctly.")
    rng = np.random.default_rng(RANDOM_STATE)
    n = 200
    # Five clusters: (income, spending) centroids
    cluster_params = [
        (87, 82, 33), (87, 18, 41), (55, 50, 43),
        (26, 78, 25), (26, 19, 45),
    ]
    rows = []
    for inc_c, sp_c, age_c in cluster_params:
        for _ in range(40):
            rows.append({
                "CustomerID": len(rows) + 1,
                "Gender": rng.choice(["Male", "Female"]),
                "Age": int(np.clip(rng.normal(age_c, 8), 18, 70)),
                "Annual Income (k$)": int(np.clip(rng.normal(inc_c, 12), 15, 137)),
                "Spending Score (1-100)": int(np.clip(rng.normal(sp_c, 10), 1, 99)),
            })
    df = pd.DataFrame(rows)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive statistics:")
print(df.describe())


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Exploratory Data Analysis — Mall Customers Dataset", fontsize=14, fontweight="bold")

# 2-a  Gender distribution
gender_counts = df["Gender"].value_counts()
axes[0, 0].bar(gender_counts.index, gender_counts.values, color=["#2196F3", "#F44336"], edgecolor="white")
axes[0, 0].set_title("Gender Distribution")
axes[0, 0].set_ylabel("Count")
for bar, val in zip(axes[0, 0].patches, gender_counts.values):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", fontsize=11, fontweight="bold")

# 2-b  Age histogram
axes[0, 1].hist(df["Age"], bins=15, color="#4CAF50", edgecolor="white")
axes[0, 1].set_title("Age Distribution")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Frequency")

# 2-c  Annual Income histogram
axes[0, 2].hist(df["Annual Income (k$)"], bins=15, color="#FF9800", edgecolor="white")
axes[0, 2].set_title("Annual Income Distribution")
axes[0, 2].set_xlabel("Annual Income (k$)")
axes[0, 2].set_ylabel("Frequency")

# 2-d  Spending Score histogram
axes[1, 0].hist(df["Spending Score (1-100)"], bins=15, color="#9C27B0", edgecolor="white")
axes[1, 0].set_title("Spending Score Distribution")
axes[1, 0].set_xlabel("Spending Score")
axes[1, 0].set_ylabel("Frequency")

# 2-e  Income vs Spending scatter
scatter = axes[1, 1].scatter(
    df["Annual Income (k$)"], df["Spending Score (1-100)"],
    c=df["Age"], cmap="viridis", alpha=0.7, edgecolors="white", linewidth=0.5, s=60
)
plt.colorbar(scatter, ax=axes[1, 1], label="Age")
axes[1, 1].set_title("Income vs Spending Score (colour = Age)")
axes[1, 1].set_xlabel("Annual Income (k$)")
axes[1, 1].set_ylabel("Spending Score")

# 2-f  Correlation heatmap
num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[1, 2], square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("01_eda.png", bbox_inches="tight")
plt.show()
print("[Saved] 01_eda.png")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: DATA PREPROCESSING")
print("=" * 60)

# Drop CustomerID (no predictive value)
df.drop(columns=["CustomerID"], inplace=True)

# Encode Gender: Male=0, Female=1
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
print("Gender encoded — unique values:", df["Gender"].unique())

# Feature matrix
X = df.copy()
feature_names = X.columns.tolist()
print("Features used for clustering:", feature_names)

# Standardize: zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled feature means (should be ~0):", X_scaled.mean(axis=0).round(4))
print("Scaled feature stds  (should be ~1):", X_scaled.std(axis=0).round(4))


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4 — DIMENSIONALITY REDUCTION (PCA)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: PCA FOR VISUALIZATION")
print("=" * 60)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print(f"PC1 explained variance: {explained[0]:.2%}")
print(f"PC2 explained variance: {explained[1]:.2%}")
print(f"Total explained:        {explained.sum():.2%}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, color="#607D8B", edgecolors="white", s=60)
ax.set_title("PCA Projection (Before Clustering)", fontsize=13, fontweight="bold")
ax.set_xlabel(f"PC 1 ({explained[0]:.1%} variance)")
ax.set_ylabel(f"PC 2 ({explained[1]:.1%} variance)")
plt.tight_layout()
plt.savefig("02_pca_raw.png", bbox_inches="tight")
plt.show()
print("[Saved] 02_pca_raw.png")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 5 — OPTIMAL K SELECTION (ELBOW + SILHOUETTE)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5: OPTIMAL K SELECTION")
print("=" * 60)

K_RANGE = range(2, 11)
wcss_list, silhouette_list = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    wcss_list.append(km.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, labels))
    print(f"  K={k:2d}  |  WCSS={km.inertia_:8.1f}  |  Silhouette={silhouette_list[-1]:.4f}")

best_k = K_RANGE.start + int(np.argmax(silhouette_list))
print(f"\nBest K by Silhouette Score: K={best_k}  (SS={max(silhouette_list):.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
ax1.plot(list(K_RANGE), wcss_list, "bo-", linewidth=2, markersize=7)
ax1.axvline(best_k, color="red", linestyle="--", linewidth=1.5, label=f"K={best_k}")
ax1.set_title("Elbow Method — WCSS vs K", fontsize=13, fontweight="bold")
ax1.set_xlabel("Number of Clusters K")
ax1.set_ylabel("Inertia (WCSS)")
ax1.legend()

# Silhouette plot
ax2.plot(list(K_RANGE), silhouette_list, "gs-", linewidth=2, markersize=7)
ax2.axvline(best_k, color="red", linestyle="--", linewidth=1.5, label=f"K={best_k}")
ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight="bold")
ax2.set_xlabel("Number of Clusters K")
ax2.set_ylabel("Silhouette Score")
ax2.legend()

plt.tight_layout()
plt.savefig("03_elbow_silhouette.png", bbox_inches="tight")
plt.show()
print("[Saved] 03_elbow_silhouette.png")

K_OPT = best_k   # use throughout the rest of the script


# ─────────────────────────────────────────────────────────────────────────
# SECTION 6 — CLUSTERING ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"SECTION 6: CLUSTERING  (K={K_OPT})")
print("=" * 60)

# ── 6-A  K-Means ──────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=K_OPT, init="k-means++", n_init=10, random_state=RANDOM_STATE)
km_labels = kmeans.fit_predict(X_scaled)
print(f"\nK-Means cluster sizes: {np.bincount(km_labels)}")

# ── 6-B  Agglomerative Hierarchical Clustering ────────────────────────────
agg = AgglomerativeClustering(n_clusters=K_OPT, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)
print(f"Agglomerative cluster sizes: {np.bincount(agg_labels)}")

# ── 6-C  DBSCAN ───────────────────────────────────────────────────────────
dbscan = DBSCAN(eps=0.5, min_samples=8)
db_labels_raw = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(db_labels_raw)) - (1 if -1 in db_labels_raw else 0)
n_noise_db    = (db_labels_raw == -1).sum()
print(f"DBSCAN  clusters found: {n_clusters_db}  |  noise points: {n_noise_db}")

# Dendrogram for Hierarchical Clustering
print("\nGenerating dendrogram ...")
Z = linkage(X_scaled, method="ward")
fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
           leaf_rotation=45, leaf_font_size=8,
           color_threshold=Z[-K_OPT, 2])
ax.axhline(y=Z[-K_OPT, 2], color="red", linestyle="--", linewidth=1.5,
           label=f"Cut for K={K_OPT}")
ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold")
ax.set_xlabel("Sample Index (truncated)")
ax.set_ylabel("Distance")
ax.legend()
plt.tight_layout()
plt.savefig("04_dendrogram.png", bbox_inches="tight")
plt.show()
print("[Saved] 04_dendrogram.png")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 7 — EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7: EVALUATION METRICS")
print("=" * 60)

def evaluate(name, X, labels):
    """Compute SS, DBI, CHI. Skip if fewer than 2 valid clusters."""
    valid = labels[labels >= 0]
    X_valid = X[labels >= 0]
    n_cl = len(set(valid))
    if n_cl < 2:
        print(f"  {name:<20s}  — fewer than 2 clusters, skipping metrics.")
        return None, None, None
    ss  = silhouette_score(X_valid, valid)
    dbi = davies_bouldin_score(X_valid, valid)
    chi = calinski_harabasz_score(X_valid, valid)
    return ss, dbi, chi

results = {}
for algo_name, labels in [
    ("K-Means",          km_labels),
    ("Agglomerative HC", agg_labels),
    ("DBSCAN",           db_labels_raw),
]:
    ss, dbi, chi = evaluate(algo_name, X_scaled, labels)
    results[algo_name] = {"Silhouette (↑)": ss, "Davies-Bouldin (↓)": dbi, "Calinski-Harabasz (↑)": chi}

metrics_df = pd.DataFrame(results).T
print("\nEvaluation Metrics Summary:")
print(metrics_df.to_string(float_format="{:.4f}".format))

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Clustering Algorithm Comparison", fontsize=14, fontweight="bold")

metric_configs = [
    ("Silhouette (↑)",       "higher is better", "#2196F3", True),
    ("Davies-Bouldin (↓)",   "lower is better",  "#F44336", False),
    ("Calinski-Harabasz (↑)","higher is better", "#4CAF50", True),
]
algos = list(results.keys())
colors_bar = ["#2196F3", "#4CAF50", "#FF9800"]

for ax, (metric, note, color, higher_better) in zip(axes, metric_configs):
    vals = [results[a].get(metric) for a in algos]
    bars = ax.bar(algos, vals, color=colors_bar, edgecolor="white", width=0.5)
    ax.set_title(f"{metric}\n({note})", fontsize=11, fontweight="bold")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, val in zip(bars, vals):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max([v for v in vals if v]) * 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    # Highlight best
    best_idx = (int(np.argmax(vals)) if higher_better
                else int(np.argmin([v if v is not None else 1e9 for v in vals])))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

plt.tight_layout()
plt.savefig("05_metrics_comparison.png", bbox_inches="tight")
plt.show()
print("[Saved] 05_metrics_comparison.png")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 8 — CLUSTER VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8: CLUSTER VISUALIZATION")
print("=" * 60)

def plot_clusters_pca(X_pca, labels, title, ax, palette, show_noise=False):
    unique = sorted(set(labels))
    for i, cl in enumerate(unique):
        mask = labels == cl
        lbl  = f"Noise" if cl == -1 else f"Cluster {cl}"
        clr  = "gray" if cl == -1 else palette[i % len(palette)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=clr, label=lbl, alpha=0.75,
                   edgecolors="white", linewidth=0.4, s=60)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8, loc="best")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Cluster Results — PCA Projection", fontsize=14, fontweight="bold")

plot_clusters_pca(X_pca, km_labels,      f"K-Means (K={K_OPT})",          axes[0], PALETTE)
plot_clusters_pca(X_pca, agg_labels,     f"Agglomerative HC (K={K_OPT})", axes[1], PALETTE)
plot_clusters_pca(X_pca, db_labels_raw,  "DBSCAN",                         axes[2], PALETTE, show_noise=True)

plt.tight_layout()
plt.savefig("06_cluster_pca.png", bbox_inches="tight")
plt.show()
print("[Saved] 06_cluster_pca.png")

# ── Income vs Spending Score scatter (K-Means) ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for cl in range(K_OPT):
    mask = km_labels == cl
    ax.scatter(df.loc[mask, "Annual Income (k$)"],
               df.loc[mask, "Spending Score (1-100)"],
               c=PALETTE[cl], label=f"Cluster {cl}", s=70,
               edgecolors="white", linewidth=0.5, alpha=0.85)
# Plot centroids (inverse-transform from scaled space)
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
# Annual Income is column index 2, Spending Score is column index 3 after dropping CustomerID & encoding
inc_idx = feature_names.index("Annual Income (k$)")
sp_idx  = feature_names.index("Spending Score (1-100)")
ax.scatter(centroids_orig[:, inc_idx], centroids_orig[:, sp_idx],
           c="black", marker="X", s=200, zorder=5, label="Centroids")
ax.set_title(f"K-Means Clusters — Income vs Spending Score (K={K_OPT})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1–100)")
ax.legend()
plt.tight_layout()
plt.savefig("07_kmeans_income_spending.png", bbox_inches="tight")
plt.show()
print("[Saved] 07_kmeans_income_spending.png")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 9 — CLUSTER PROFILING & INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 9: CLUSTER PROFILING")
print("=" * 60)

df_profile = df.copy()
df_profile["Cluster"] = km_labels

# Per-cluster mean
profile = df_profile.groupby("Cluster").agg(
    Count=("Cluster", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
    Pct_Female=("Gender", "mean"),   # Gender: Female=1
).round(2)

CLUSTER_LABELS = {
    0: "High-Inc, High-Spend (Premium)",
    1: "High-Inc, Low-Spend  (Conservative)",
    2: "Moderate All-Rounder",
    3: "Low-Inc,  High-Spend  (Impulsive)",
    4: "Low-Inc,  Low-Spend   (Budget)",
}

# Map labels to profile (sort by income descending to assign label names robustly)
profile_sorted = profile.sort_values("Avg_Income", ascending=False).copy()
label_list = [
    "High-Inc, High-Spend (Premium)",
    "High-Inc, Low-Spend  (Conservative)",
    "Moderate All-Rounder",
    "Low-Inc,  High-Spend  (Impulsive)",
    "Low-Inc,  Low-Spend   (Budget)",
]
# Sort by income then spending for a consistent label assignment
profile_sorted2 = profile.copy()
profile_sorted2["Label"] = ""
sorted_idx = profile.sort_values(["Avg_Income", "Avg_Spending"], ascending=[False, False]).index

# Simple heuristic assignment using thresholds
for cl in profile.index:
    inc  = profile.loc[cl, "Avg_Income"]
    sp   = profile.loc[cl, "Avg_Spending"]
    if inc >= 70 and sp >= 60:
        lbl = "High-Inc, High-Spend (Premium)"
    elif inc >= 70 and sp < 40:
        lbl = "High-Inc, Low-Spend (Conservative)"
    elif inc < 40 and sp >= 60:
        lbl = "Low-Inc, High-Spend (Impulsive)"
    elif inc < 40 and sp < 40:
        lbl = "Low-Inc, Low-Spend (Budget)"
    else:
        lbl = "Moderate All-Rounder"
    profile_sorted2.loc[cl, "Label"] = lbl

print("\nCluster Profiles:")
print(profile_sorted2[["Count", "Avg_Age", "Avg_Income", "Avg_Spending", "Pct_Female", "Label"]].to_string())

# Radar / Spider chart for cluster profiles
categories = ["Age", "Income (k$)", "Spending Score", "% Female"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)

# Normalize each feature to [0, 1] for the radar
norm_data = profile[["Avg_Age", "Avg_Income", "Avg_Spending", "Pct_Female"]].copy()
for col in norm_data.columns:
    mn, mx = norm_data[col].min(), norm_data[col].max()
    norm_data[col] = (norm_data[col] - mn) / (mx - mn + 1e-9)

for cl in profile.index:
    vals = norm_data.loc[cl].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, color=PALETTE[cl % len(PALETTE)],
            label=f"Cluster {cl}")
    ax.fill(angles, vals, alpha=0.1, color=PALETTE[cl % len(PALETTE)])

ax.set_title("Cluster Profiles — Radar Chart\n(values normalized to [0,1])",
             size=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("08_radar_chart.png", bbox_inches="tight")
plt.show()
print("[Saved] 08_radar_chart.png")

# Bar chart: cluster size
fig, ax = plt.subplots(figsize=(8, 4))
cluster_sizes = np.bincount(km_labels)
bars = ax.bar(range(K_OPT), cluster_sizes, color=PALETTE[:K_OPT], edgecolor="white")
ax.set_title("K-Means Cluster Sizes", fontsize=13, fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Customers")
ax.set_xticks(range(K_OPT))
for bar, val in zip(bars, cluster_sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("09_cluster_sizes.png", bbox_inches="tight")
plt.show()
print("[Saved] 09_cluster_sizes.png")


# ─────────────────────────────────────────────────────────────────────────
# SECTION 10 — SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 10: FINAL SUMMARY REPORT")
print("=" * 60)

print(f"""
Project : Customer Segmentation — ML Stage 2
Student : Mohammed Saad  (F20230214)
Dataset : Mall Customers (n=200, 5 features)
GitHub  : https://github.com/MohammedSaad-ML/CustomerSegmentation-ML

── Algorithm Comparison (K={K_OPT}) ─────────────────────────────────
{metrics_df.to_string(float_format="{:.4f}".format)}

── Best Algorithm ──────────────────────────────────────────────────
  K-Means (K={K_OPT})
  Silhouette Score  = {results['K-Means']['Silhouette (↑)']:.4f}  (higher is better, max=1)
  Davies-Bouldin    = {results['K-Means']['Davies-Bouldin (↓)']:.4f}  (lower is better)
  Calinski-Harabasz = {results['K-Means']['Calinski-Harabasz (↑)']:.2f} (higher is better)

── Cluster Profiles ────────────────────────────────────────────────
{profile_sorted2[['Count','Avg_Age','Avg_Income','Avg_Spending','Label']].to_string()}

── Marketing Recommendations ───────────────────────────────────────
  Premium Buyers     → Loyalty rewards, premium product lines, VIP events
  Conservative Wealthy → Trust-based messaging, value-quality emphasis
  Moderate           → Bundle deals, mid-range promotions
  Impulsive Buyers   → Trend campaigns, discounts, installment plans
  Budget-Conscious   → Essential goods offers, price-match guarantees

── Output Files Saved ──────────────────────────────────────────────
  01_eda.png                  Exploratory Data Analysis
  02_pca_raw.png              PCA Projection (raw)
  03_elbow_silhouette.png     Elbow + Silhouette Method
  04_dendrogram.png           Hierarchical Dendrogram
  05_metrics_comparison.png   Algorithm Metric Comparison
  06_cluster_pca.png          All 3 Algorithms — PCA Clusters
  07_kmeans_income_spending.png  K-Means: Income vs Spending
  08_radar_chart.png          Cluster Profiles Radar Chart
  09_cluster_sizes.png        Cluster Size Bar Chart
""")


