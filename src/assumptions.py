"""
Step E: classical OLS regression-assumption diagnostics, run on Scenario A
(G1+G2 included) since that is the specification where a linear model is
actually a good fit (R2 ~0.82) and coefficients are worth trusting.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

from viz_style import apply_style, savefig, CAT, INK_SECONDARY

apply_style()
FIGDIR = "../outputs/figures"

df = pd.read_csv("../data/student-mat.csv", sep=";")

# Compact, interpretable feature set for the OLS assumption check
# (avoid the full one-hot blow-up so VIF / coefficients stay readable)
df["higher_bin"] = (df["higher"] == "yes").astype(int)
df["internet_bin"] = (df["internet"] == "yes").astype(int)
df["schoolsup_bin"] = (df["schoolsup"] == "yes").astype(int)

features = ["G1", "G2", "failures", "studytime", "absences", "Medu", "Fedu",
            "age", "goout", "higher_bin", "internet_bin", "schoolsup_bin"]
X = df[features].astype(float)
y = df["G3"].astype(float)

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
resid = model.resid
fitted = model.fittedvalues

report = {}
report["r_squared"] = model.rsquared
report["adj_r_squared"] = model.rsquared_adj
report["f_pvalue"] = model.f_pvalue
report["coefficients"] = model.params.round(4).to_dict()
report["p_values"] = model.pvalues.round(4).to_dict()

# ---- 1. Linearity: correlation of |resid| with fitted, and per-predictor
# component check is visual (see fig). Rainbow test as a formal linearity test.
from statsmodels.stats.diagnostic import linear_rainbow
rainbow_stat, rainbow_p = linear_rainbow(model)
report["linearity_rainbow_test"] = {"stat": rainbow_stat, "p_value": rainbow_p,
                                     "interpretation": "p>0.05 fails to reject linearity" }

# ---- 2. Normality of residuals ----
shapiro_stat, shapiro_p = stats.shapiro(resid)
jb_stat, jb_p, skew, kurt = sm.stats.jarque_bera(resid)
report["normality"] = {
    "shapiro_stat": shapiro_stat, "shapiro_p": shapiro_p,
    "jarque_bera_stat": jb_stat, "jarque_bera_p": jb_p,
    "skew": skew, "kurtosis": kurt,
    "interpretation": "p<0.05 => reject normality of residuals"
}

# ---- 3. Homoscedasticity: Breusch-Pagan ----
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(resid, X_const)
report["homoscedasticity_breusch_pagan"] = {
    "lm_stat": bp_stat, "lm_p": bp_p,
    "interpretation": "p<0.05 => reject homoscedasticity (heteroscedastic)"
}

# ---- 4. Multicollinearity: VIF ----
vif_data = pd.DataFrame()
vif_data["feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
report["vif"] = vif_data.set_index("feature")["VIF"].round(2).to_dict()

# ---- 5. Independence of errors: Durbin-Watson ----
dw = durbin_watson(resid)
report["durbin_watson"] = {"stat": dw,
                            "interpretation": "~2 => no autocorrelation (caveat: no natural ordering in this cross-sectional data, test kept for completeness)"}

# ---- 6. Influential points: Cook's distance ----
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
n = len(y)
threshold = 4 / n
n_influential = int((cooks_d > threshold).sum())
report["influential_points"] = {"threshold_4_over_n": threshold, "n_flagged": n_influential,
                                 "pct_flagged": round(100 * n_influential / n, 2)}

with open("../outputs/assumptions_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

# ================= PLOTS =================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].scatter(fitted, resid, alpha=0.5, s=24, color=CAT[0], edgecolor="none")
axes[0, 0].axhline(0, color=CAT[7], linestyle="--", linewidth=2)
axes[0, 0].set_xlabel("Fitted values"); axes[0, 0].set_ylabel("Residuals")
axes[0, 0].set_title("Residuals vs Fitted (Linearity / Homoscedasticity)")

stats.probplot(resid, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[0].set_markerfacecolor(CAT[0]); axes[0, 1].get_lines()[0].set_markeredgecolor(CAT[0])
axes[0, 1].get_lines()[1].set_color(CAT[7])
axes[0, 1].set_title(f"Normal Q-Q (Shapiro p={shapiro_p:.4f})")

axes[1, 0].hist(resid, bins=30, color=CAT[2], edgecolor="white", linewidth=0.5)
axes[1, 0].set_title("Residual Distribution")
axes[1, 0].set_xlabel("Residual")

sqrt_abs_resid = np.sqrt(np.abs(resid / resid.std()))
axes[1, 1].scatter(fitted, sqrt_abs_resid, alpha=0.5, s=24, color=CAT[3], edgecolor="none")
axes[1, 1].set_xlabel("Fitted values"); axes[1, 1].set_ylabel("sqrt(|standardized resid|)")
axes[1, 1].set_title(f"Scale-Location (Breusch-Pagan p={bp_p:.4f})")

savefig(fig, f"{FIGDIR}/08_ols_assumption_diagnostics.png")

fig, ax = plt.subplots(figsize=(8, 4.4))
vif_plot = vif_data[vif_data["feature"] != "const"].sort_values("VIF")
colors = [CAT[7] if v > 5 else CAT[0] for v in vif_plot["VIF"]]
ax.barh(vif_plot["feature"], vif_plot["VIF"], color=colors)
ax.axvline(5, color=CAT[3], linestyle="--", linewidth=1.5, label="VIF=5 caution")
ax.set_title("Variance Inflation Factor per Predictor")
ax.set_xlabel("VIF")
ax.legend(frameon=False)
savefig(fig, f"{FIGDIR}/09_vif.png")

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.stem(np.arange(n), cooks_d, markerfmt=",", basefmt=" ")
ax.axhline(threshold, color=CAT[7], linestyle="--", linewidth=1.5, label=f"4/n = {threshold:.3f}")
ax.set_title(f"Cook's Distance ({n_influential} students flagged, {report['influential_points']['pct_flagged']}%)")
ax.set_xlabel("Observation index"); ax.set_ylabel("Cook's D")
ax.legend(frameon=False)
savefig(fig, f"{FIGDIR}/10_cooks_distance.png")

print(json.dumps(report, indent=2, default=str))
