"""Step A/D: End-to-end data understanding + EDA with visualizations."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from viz_style import apply_style, savefig, CAT, BLUE_SEQ, INK_SECONDARY, INK_MUTED, GRID

apply_style()

DATA = "../data/student-mat.csv"
FIGDIR = "../outputs/figures"

df = pd.read_csv(DATA, sep=";")
insights = {}

insights["shape"] = df.shape
insights["dtypes"] = df.dtypes.astype(str).to_dict()
insights["missing_total"] = int(df.isna().sum().sum())
insights["duplicates"] = int(df.duplicated().sum())
insights["target_stats"] = df["G3"].describe().to_dict()
insights["zero_G3_count"] = int((df["G3"] == 0).sum())
insights["pct_zero_G3"] = round(100 * (df["G3"] == 0).mean(), 2)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
insights["numeric_cols"] = numeric_cols
insights["categorical_cols"] = cat_cols

corr_with_target = df[numeric_cols].corr()["G3"].drop("G3").sort_values(key=lambda s: -s.abs())
insights["top_corr_with_G3"] = corr_with_target.round(3).to_dict()

# ---------- Fig 1: Target distribution ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(df["G3"], bins=range(0, 22), color=CAT[0], edgecolor="white", linewidth=0.6)
ax.axvline(df["G3"].mean(), color=CAT[7], linestyle="--", linewidth=2, label=f"mean = {df['G3'].mean():.1f}")
ax.set_title("Distribution of Final Grade (G3)")
ax.set_xlabel("G3 (0-20)")
ax.set_ylabel("Number of students")
ax.legend(frameon=False)
savefig(fig, f"{FIGDIR}/01_target_distribution.png")

# ---------- Fig 2: G1, G2, G3 relationship ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
pairs = [("G1", "G3"), ("G2", "G3"), ("G1", "G2")]
for ax, (x, y) in zip(axes, pairs):
    ax.scatter(df[x], df[y], alpha=0.45, s=22, color=CAT[0], edgecolor="none")
    r = df[x].corr(df[y])
    ax.set_title(f"{x} vs {y}  (r={r:.2f})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
savefig(fig, f"{FIGDIR}/02_grade_progression.png")

# ---------- Fig 3: Correlation heatmap ----------
fig, ax = plt.subplots(figsize=(9, 7.5))
corr = df[numeric_cols].corr()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=90, fontsize=8)
ax.set_yticks(range(len(numeric_cols)))
ax.set_yticklabels(numeric_cols, fontsize=8)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        v = corr.iloc[i, j]
        if abs(v) >= 0.3:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                     color="white" if abs(v) > 0.6 else INK_SECONDARY)
ax.set_title("Correlation Matrix — Numeric Features")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
savefig(fig, f"{FIGDIR}/03_correlation_heatmap.png")

# ---------- Fig 4: Categorical breakdowns vs G3 ----------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
cat_specs = ["school", "sex", "address", "internet", "romantic", "higher"]
for ax, col in zip(axes.flat, cat_specs):
    groups = df.groupby(col)["G3"].mean().sort_values(ascending=False)
    bars = ax.bar(groups.index.astype(str), groups.values, color=CAT[0], width=0.55)
    ax.set_title(f"Mean G3 by {col}")
    ax.set_ylabel("Mean G3")
    for b, v in zip(bars, groups.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}", ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylim(0, max(groups.values) + 2.5)
savefig(fig, f"{FIGDIR}/04_categorical_breakdowns.png")

# ---------- Fig 5: Key numeric drivers vs G3 (boxplots) ----------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
box_specs = ["failures", "studytime", "goout", "Walc", "Dalc", "famrel"]
for ax, col in zip(axes.flat, box_specs):
    groups = [df.loc[df[col] == v, "G3"].values for v in sorted(df[col].unique())]
    bp = ax.boxplot(groups, tick_labels=sorted(df[col].unique()), patch_artist=True, widths=0.55,
                     medianprops=dict(color=CAT[7], linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(CAT[0])
        patch.set_alpha(0.55)
        patch.set_edgecolor(CAT[0])
    ax.set_title(f"G3 by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("G3")
savefig(fig, f"{FIGDIR}/05_numeric_drivers_boxplots.png")

# ---------- Fig 6: Absences vs G3 ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(df["absences"], df["G3"], alpha=0.45, s=22, color=CAT[2], edgecolor="none")
z = np.polyfit(df["absences"], df["G3"], 1)
xs = np.linspace(df["absences"].min(), df["absences"].max(), 50)
ax.plot(xs, np.polyval(z, xs), color=CAT[7], linewidth=2, linestyle="--")
ax.set_title(f"Absences vs G3 (r={df['absences'].corr(df['G3']):.2f})")
ax.set_xlabel("Absences")
ax.set_ylabel("G3")
savefig(fig, f"{FIGDIR}/06_absences_vs_g3.png")

# ---------- Fig 7: Parental education & job ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
pe = df.groupby("Medu")["G3"].mean()
axes[0].bar(pe.index.astype(str), pe.values, color=CAT[0])
axes[0].set_title("Mean G3 by Mother's Education")
axes[0].set_xlabel("Medu (0=none .. 4=higher)")
axes[0].set_ylabel("Mean G3")

mj = df.groupby("Mjob")["G3"].mean().sort_values(ascending=False)
axes[1].bar(mj.index.astype(str), mj.values, color=CAT[2])
axes[1].set_title("Mean G3 by Mother's Job")
axes[1].set_ylabel("Mean G3")
savefig(fig, f"{FIGDIR}/07_parent_education_job.png")

with open("../outputs/eda_insights.json", "w") as f:
    json.dump(insights, f, indent=2, default=str)

print("EDA complete.")
print(json.dumps(insights, indent=2, default=str)[:3000])
