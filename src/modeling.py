"""
Steps B/C: preprocessing, feature engineering, feature selection,
train/test split, model selection, hyperparameter tuning, evaluation.

Two scenarios are run (standard in the Cortez & Silva UCI paper):
  - Scenario A ("with prior grades"): G1 & G2 included as predictors.
  - Scenario B ("early prediction"): G1 & G2 EXCLUDED -> the genuinely
    hard / realistic ML problem (predict final grade from behavioural &
    demographic data only, before any exam has been sat).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from viz_style import apply_style, savefig, CAT, INK_SECONDARY

apply_style()
RANDOM_STATE = 42
FIGDIR = "../outputs/figures"

df = pd.read_csv("../data/student-mat.csv", sep=";")

# ----------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ----------------------------------------------------------------------
df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2
df["alc_total"] = df["Dalc"] * 5 + df["Walc"] * 2          # weekly-weighted alcohol use
df["support_score"] = (df[["schoolsup", "famsup", "paid"]] == "yes").sum(axis=1)
df["study_per_failure"] = df["studytime"] / (df["failures"] + 1)
df["social_score"] = df["freetime"] + df["goout"]
df["log_absences"] = np.log1p(df["absences"])              # absences is right-skewed

binary_map_cols = ["school", "sex", "address", "famsize", "Pstatus",
                    "schoolsup", "famsup", "paid", "activities", "nursery",
                    "higher", "internet", "romantic"]
binary_maps = {
    "school": {"GP": 0, "MS": 1}, "sex": {"F": 0, "M": 1},
    "address": {"U": 0, "R": 1}, "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
}
yn_cols = ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
for c in yn_cols:
    binary_maps[c] = {"no": 0, "yes": 1}

for c in binary_map_cols:
    df[c] = df[c].map(binary_maps[c])

nominal_cols = ["Mjob", "Fjob", "reason", "guardian"]

engineered_num = ["parent_edu_avg", "alc_total", "support_score", "study_per_failure",
                   "social_score", "log_absences"]
base_num = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
            "freetime", "goout", "Dalc", "Walc", "health", "absences"]

results_all = {}


def build_preprocessor(numeric_features, nominal_features, binary_features):
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("nom", OneHotEncoder(handle_unknown="ignore", drop="first"), nominal_features),
        ("bin", "passthrough", binary_features),
    ])


def run_scenario(name, extra_cols):
    numeric_features = base_num + engineered_num
    binary_features = binary_map_cols
    nominal_features = nominal_cols
    feature_cols = numeric_features + binary_features + nominal_features + extra_cols
    numeric_features = numeric_features + extra_cols  # G1/G2 treated as numeric if present

    X = df[numeric_features + binary_features + nominal_features]
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    pre = build_preprocessor(numeric_features, nominal_features, binary_features)

    # ---- feature selection diagnostics (on train, transformed) ----
    pre_fit = pre.fit(X_train)
    Xt_train = pre_fit.transform(X_train)
    feat_names = (numeric_features
                  + list(pre_fit.named_transformers_["nom"].get_feature_names_out(nominal_features))
                  + binary_features)
    skb = SelectKBest(score_func=f_regression, k="all").fit(Xt_train, y_train)
    fscores = pd.Series(skb.scores_, index=feat_names).sort_values(ascending=False)

    rf_fs = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE).fit(Xt_train, y_train)
    importances = pd.Series(rf_fs.feature_importances_, index=feat_names).sort_values(ascending=False)

    # ---- model zoo ----
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=5, epsilon=0.3),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = {}
    for mname, mdl in models.items():
        pipe = Pipeline([("pre", pre), ("model", mdl)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        rmse_scores = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        cv_summary[mname] = {"cv_r2_mean": scores.mean(), "cv_r2_std": scores.std(),
                              "cv_rmse_mean": rmse_scores.mean()}

    cv_df = pd.DataFrame(cv_summary).T.sort_values("cv_r2_mean", ascending=False)

    # ---- hyperparameter tuning of top 3 by CV R2 ----
    param_grids = {
        "Ridge": {"model__alpha": [0.1, 1.0, 5.0, 10.0, 30.0]},
        "Lasso": {"model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
        "ElasticNet": {"model__alpha": [0.01, 0.1, 0.5, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
        "KNN": {"model__n_neighbors": [3, 5, 7, 9, 11, 15]},
        "DecisionTree": {"model__max_depth": [3, 4, 5, 6, 8, None]},
        "RandomForest": {"model__n_estimators": [200, 400], "model__max_depth": [None, 5, 8, 12],
                          "model__min_samples_leaf": [1, 2, 4]},
        "GradientBoosting": {"model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 4],
                              "model__learning_rate": [0.03, 0.05, 0.1]},
        "SVR": {"model__C": [1, 5, 10, 20], "model__epsilon": [0.1, 0.3, 0.5], "model__gamma": ["scale", "auto"]},
        "LinearRegression": {},
    }

    top3 = cv_df.head(3).index.tolist()
    tuned_results = {}
    best_estimators = {}
    for mname in top3:
        pipe = Pipeline([("pre", pre), ("model", models[mname])])
        grid = param_grids.get(mname, {})
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="r2", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            best_params = gs.best_params_
        else:
            best_pipe = pipe.fit(X_train, y_train)
            best_params = {}
        y_pred = best_pipe.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        nonzero = y_test != 0
        mape = (np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])).mean() * 100
        tuned_results[mname] = {"best_params": best_params, "test_r2": r2, "test_rmse": rmse,
                                 "test_mae": mae, "test_mape": mape}
        best_estimators[mname] = best_pipe

    best_name = max(tuned_results, key=lambda k: tuned_results[k]["test_r2"])
    best_pipe = best_estimators[best_name]
    y_pred_best = best_pipe.predict(X_test)

    # ---- plots: model comparison, actual vs predicted, residuals, importances ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    order = cv_df.index
    ax.barh(order, cv_df["cv_r2_mean"], xerr=cv_df["cv_r2_std"], color=CAT[0], height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV R² (mean ± std)")
    ax.set_title(f"Model comparison — {name}")
    savefig(fig, f"{FIGDIR}/{name}_model_comparison.png")

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(y_test, y_pred_best, alpha=0.55, s=28, color=CAT[0], edgecolor="none")
    lims = [min(y_test.min(), y_pred_best.min()) - 1, max(y_test.max(), y_pred_best.max()) + 1]
    ax.plot(lims, lims, color=CAT[7], linestyle="--", linewidth=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual G3"); ax.set_ylabel("Predicted G3")
    ax.set_title(f"Actual vs Predicted — {best_name} ({name})\nR²={tuned_results[best_name]['test_r2']:.3f}")
    savefig(fig, f"{FIGDIR}/{name}_actual_vs_predicted.png")

    residuals = y_test - y_pred_best
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(y_pred_best, residuals, alpha=0.55, s=26, color=CAT[2], edgecolor="none")
    axes[0].axhline(0, color=CAT[7], linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted G3"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor(CAT[0])
    axes[1].get_lines()[0].set_markeredgecolor(CAT[0])
    axes[1].get_lines()[1].set_color(CAT[7])
    axes[1].set_title("Q-Q Plot of Residuals")
    savefig(fig, f"{FIGDIR}/{name}_residual_diagnostics.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    top_imp = importances.head(12).sort_values()
    ax.barh(top_imp.index, top_imp.values, color=CAT[2])
    ax.set_title(f"Random Forest Feature Importance — {name}")
    ax.set_xlabel("Importance")
    savefig(fig, f"{FIGDIR}/{name}_feature_importance.png")

    results_all[name] = {
        "feature_cols": feature_cols,
        "n_train": len(X_train), "n_test": len(X_test),
        "top_fscore_features": fscores.head(10).round(1).to_dict(),
        "top_rf_importance_features": importances.head(10).round(4).to_dict(),
        "cv_summary": cv_df.round(4).to_dict(orient="index"),
        "tuned_results": {k: {kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()}
                           for k, v in tuned_results.items()},
        "best_model": best_name,
        "best_test_metrics": tuned_results[best_name],
    }
    return results_all[name]


print("Running Scenario A (with G1 & G2)...")
run_scenario("scenarioA_with_grades", extra_cols=["G1", "G2"])

print("Running Scenario B (early prediction, no G1/G2)...")
run_scenario("scenarioB_early_prediction", extra_cols=[])

with open("../outputs/model_results.json", "w") as f:
    json.dump(results_all, f, indent=2, default=str)

print(json.dumps({k: v["best_model"] for k, v in results_all.items()}, indent=2))
print(json.dumps({k: v["best_test_metrics"] for k, v in results_all.items()}, indent=2))
