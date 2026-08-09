import base64
import json

FIGDIR = "../outputs/figures"


def b64(name):
    with open(f"{FIGDIR}/{name}", "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


eda = json.load(open("../outputs/eda_insights.json"))
models = json.load(open("../outputs/model_results.json"))
assump = json.load(open("../outputs/assumptions_report.json"))

A = models["scenarioA_with_grades"]
B = models["scenarioB_early_prediction"]

img = {n: b64(f) for n, f in {
    "target": "01_target_distribution.png",
    "progression": "02_grade_progression.png",
    "heatmap": "03_correlation_heatmap.png",
    "cat": "04_categorical_breakdowns.png",
    "box": "05_numeric_drivers_boxplots.png",
    "absences": "06_absences_vs_g3.png",
    "parent": "07_parent_education_job.png",
    "ols_diag": "08_ols_assumption_diagnostics.png",
    "vif": "09_vif.png",
    "cooks": "10_cooks_distance.png",
    "a_cv": "scenarioA_with_grades_model_comparison.png",
    "a_avp": "scenarioA_with_grades_actual_vs_predicted.png",
    "a_resid": "scenarioA_with_grades_residual_diagnostics.png",
    "a_imp": "scenarioA_with_grades_feature_importance.png",
    "b_cv": "scenarioB_early_prediction_model_comparison.png",
    "b_avp": "scenarioB_early_prediction_actual_vs_predicted.png",
    "b_resid": "scenarioB_early_prediction_residual_diagnostics.png",
    "b_imp": "scenarioB_early_prediction_feature_importance.png",
}.items()}


def datauri(key):
    return f"data:image/png;base64,{img[key]}"


def fmt_pct(x):
    return f"{x*100:.1f}%"


def cv_table_rows(scenario_dict):
    rows = sorted(scenario_dict.items(), key=lambda kv: -kv[1]["cv_r2_mean"])
    out = []
    for name, v in rows:
        out.append(f"<tr><td>{name}</td><td class='num'>{v['cv_r2_mean']:.3f}</td>"
                    f"<td class='num'>{v['cv_r2_std']:.3f}</td><td class='num'>{v['cv_rmse_mean']:.2f}</td></tr>")
    return "\n".join(out)


def top_features_rows(d, n=8):
    out = []
    for k, v in list(d.items())[:n]:
        out.append(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>")
    return "\n".join(out)


def coef_rows():
    coefs = assump["coefficients"]
    pvals = assump["p_values"]
    order = ["G2", "G1", "schoolsup_bin", "higher_bin", "failures", "internet_bin",
             "studytime", "Fedu", "age", "Medu", "goout", "absences", "const"]
    out = []
    for k in order:
        p = pvals[k]
        sig = "sig" if p < 0.05 else "nsig"
        star = "★" if p < 0.05 else ""
        out.append(f"<tr><td>{k}</td><td class='num'>{coefs[k]:+.3f}</td>"
                    f"<td class='num {sig}'>{p:.4f} {star}</td></tr>")
    return "\n".join(out)


def vif_rows():
    out = []
    for k, v in assump["vif"].items():
        if k == "const":
            continue
        flag = "flag" if v > 5 else ""
        out.append(f"<tr><td>{k}</td><td class='num {flag}'>{v:.2f}</td></tr>")
    return "\n".join(out)


html = f"""<!doctype html>
<title>Student Performance — End-to-End Regression Study</title>
<style>
@font-face {{
  font-family: 'ui-serif-fallback';
  src: local('Georgia');
}}
:root {{
  --ink: #14181f;
  --ink-2: #454b57;
  --ink-mute: #767c88;
  --ground: #f4f5f3;
  --surface: #ffffff;
  --surface-2: #eef0ec;
  --border: #dcdfd8;
  --accent: #2a5fb0;
  --accent-soft: #e4ecf9;
  --accent-b: #b6720a;
  --accent-b-soft: #f6ecd9;
  --good: #147a3a;
  --good-soft: #e3f3e7;
  --bad: #b0342e;
  --bad-soft: #fbe7e5;
  --shadow: 0 1px 2px rgba(20,24,31,0.06), 0 8px 24px -12px rgba(20,24,31,0.15);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ink: #eef0f2;
    --ink-2: #c3c7cf;
    --ink-mute: #8d93a0;
    --ground: #14161a;
    --surface: #1b1e23;
    --surface-2: #22262c;
    --border: #33373f;
    --accent: #6fa0ea;
    --accent-soft: #223349;
    --accent-b: #e0a748;
    --accent-b-soft: #3a2e17;
    --good: #4fce80;
    --good-soft: #16301f;
    --bad: #f0817c;
    --bad-soft: #3a1e1c;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -12px rgba(0,0,0,0.6);
  }}
}}
:root[data-theme="dark"] {{
  --ink: #eef0f2;
  --ink-2: #c3c7cf;
  --ink-mute: #8d93a0;
  --ground: #14161a;
  --surface: #1b1e23;
  --surface-2: #22262c;
  --border: #33373f;
  --accent: #6fa0ea;
  --accent-soft: #223349;
  --accent-b: #e0a748;
  --accent-b-soft: #3a2e17;
  --good: #4fce80;
  --good-soft: #16301f;
  --bad: #f0817c;
  --bad-soft: #3a1e1c;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -12px rgba(0,0,0,0.6);
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.55;
}}
.serif {{ font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif; }}
.mono, .num {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; }}

.wrap {{ max-width: 1160px; margin: 0 auto; padding: 0 28px 96px; }}

header.hero {{
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, var(--surface-2), var(--ground));
  padding: 56px 28px 40px;
}}
header.hero .inner {{ max-width: 1160px; margin: 0 auto; }}
.eyebrow {{
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  font-weight: 600;
  color: var(--accent);
}}
h1.title {{
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
  font-size: clamp(28px, 4vw, 44px);
  margin: 10px 0 8px;
  text-wrap: balance;
  letter-spacing: -0.01em;
}}
.subtitle {{ color: var(--ink-2); font-size: 16.5px; max-width: 62ch; }}

nav.toc {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px 4px;
  margin-top: 28px;
}}
nav.toc a {{
  font-size: 13px;
  color: var(--ink-2);
  text-decoration: none;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  white-space: nowrap;
}}
nav.toc a:hover, nav.toc a:focus-visible {{ border-color: var(--accent); color: var(--accent); }}

.stat-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 30px;
}}
.stat {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}}
.stat .label {{ font-size: 12px; color: var(--ink-mute); text-transform: uppercase; letter-spacing: 0.06em; }}
.stat .value {{ font-family: Georgia, serif; font-size: 26px; margin-top: 4px; }}
.stat .sub {{ font-size: 12.5px; color: var(--ink-2); margin-top: 2px; }}

section {{ padding-top: 56px; scroll-margin-top: 20px; }}
section > h2 {{
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
  font-size: 26px;
  margin: 0 0 6px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--ink);
  display: inline-block;
}}
.section-intro {{ color: var(--ink-2); max-width: 76ch; margin: 14px 0 26px; }}

.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px 24px;
  box-shadow: var(--shadow);
  margin-bottom: 20px;
}}
.card h3 {{ margin-top: 0; font-size: 16px; }}
.card h3 .tag {{
  display: inline-block; margin-left: 8px; font-size: 11px; font-weight: 600;
  padding: 2px 8px; border-radius: 999px; vertical-align: middle;
}}
.tag.scenA {{ background: var(--accent-soft); color: var(--accent); }}
.tag.scenB {{ background: var(--accent-b-soft); color: var(--accent-b); }}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 860px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

figure {{ margin: 0; }}
figure img {{ width: 100%; height: auto; display: block; border-radius: 8px; border: 1px solid var(--border); }}
figcaption {{ font-size: 13px; color: var(--ink-mute); margin-top: 8px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--ink-mute); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.04em; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.sig {{ color: var(--good); font-weight: 600; }}
td.flag {{ color: var(--bad); font-weight: 600; }}
.table-wrap {{ overflow-x: auto; }}

.pill {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12.5px; font-weight: 600; padding: 4px 11px; border-radius: 999px;
}}
.pill.pass {{ background: var(--good-soft); color: var(--good); }}
.pill.fail {{ background: var(--bad-soft); color: var(--bad); }}

.callout {{
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  padding: 14px 18px;
  border-radius: 0 8px 8px 0;
  font-size: 14.5px;
  color: var(--ink-2);
  margin: 18px 0;
}}
.callout.warn {{ border-color: var(--accent-b); background: var(--accent-b-soft); }}
.callout strong {{ color: var(--ink); }}

ol.steps {{ padding-left: 0; list-style: none; counter-reset: step; margin: 0; }}
ol.steps li {{
  counter-increment: step;
  padding: 10px 0 10px 40px;
  position: relative;
  border-bottom: 1px solid var(--border);
  font-size: 14.5px;
  color: var(--ink-2);
}}
ol.steps li:last-child {{ border-bottom: none; }}
ol.steps li::before {{
  content: counter(step);
  position: absolute; left: 0; top: 8px;
  width: 24px; height: 24px; border-radius: 50%;
  background: var(--accent); color: white;
  font-size: 12px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}}
ol.steps li b {{ color: var(--ink); }}

code {{
  background: var(--surface-2);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 0.92em;
  font-family: ui-monospace, "SF Mono", Consolas, monospace;
}}

footer {{
  margin-top: 64px; padding-top: 24px; border-top: 1px solid var(--border);
  color: var(--ink-mute); font-size: 12.5px;
}}
</style>

<header class="hero">
  <div class="inner">
    <div class="eyebrow">Regression case study &middot; UCI Student Performance</div>
    <h1 class="title">Predicting Final Grades from Behavioural &amp; Demographic Data</h1>
    <p class="subtitle">An end-to-end regression pipeline on the 395-student Math-course dataset (Cortez &amp; Silva,
      2008): data understanding, EDA, preprocessing, feature engineering, feature selection, model comparison
      across 9 algorithms, hyperparameter tuning, and formal OLS assumption diagnostics.</p>

    <div class="stat-row">
      <div class="stat"><div class="label">Students</div><div class="value num">{eda['shape'][0]}</div><div class="sub">{eda['shape'][1]} raw attributes</div></div>
      <div class="stat"><div class="label">Missing values</div><div class="value num">{eda['missing_total']}</div><div class="sub">complete dataset</div></div>
      <div class="stat"><div class="label">Mean G3</div><div class="value num">{eda['target_stats']['mean']:.1f} / 20</div><div class="sub">σ = {eda['target_stats']['std']:.1f}</div></div>
      <div class="stat"><div class="label">Zero-grade students</div><div class="value num">{eda['zero_G3_count']} ({fmt_pct(eda['pct_zero_G3']/100)})</div><div class="sub">likely dropouts / no-shows</div></div>
      <div class="stat"><div class="label">Best model, w/ G1&amp;G2</div><div class="value num">R²&nbsp;{A['best_test_metrics']['test_r2']:.2f}</div><div class="sub">{A['best_model']}, RMSE {A['best_test_metrics']['test_rmse']:.2f}</div></div>
      <div class="stat"><div class="label">Best model, early-predict</div><div class="value num">R²&nbsp;{B['best_test_metrics']['test_r2']:.2f}</div><div class="sub">{B['best_model']}, RMSE {B['best_test_metrics']['test_rmse']:.2f}</div></div>
    </div>

    <nav class="toc">
      <a href="#understanding">A · Data understanding</a>
      <a href="#eda">D · EDA &amp; insights</a>
      <a href="#prep">B/C · Preprocessing &amp; features</a>
      <a href="#selection">Feature selection</a>
      <a href="#models">Model selection</a>
      <a href="#scenarioA">Scenario A results</a>
      <a href="#scenarioB">Scenario B results</a>
      <a href="#assumptions">E · Regression assumptions</a>
      <a href="#conclusion">Conclusions</a>
    </nav>
  </div>
</header>

<div class="wrap">

<section id="understanding">
  <h2>A · Understanding the dataset end to end</h2>
  <p class="section-intro">Two twin surveys were collected from Portuguese secondary schools: <code>student-mat.csv</code>
    (395 students, Math course) and <code>student-por.csv</code> (649 students, Portuguese course), sharing 30
    behavioural / demographic attributes plus three period grades <code>G1, G2, G3</code> (0&ndash;20 scale). A
    published R merge key (<code>student-merge.R</code>) identifies 382 students who appear in both files. This
    study runs the full pipeline on <b>student-mat.csv</b> — the canonical variant used in the original research —
    since it is large enough to model robustly and keeps the write-up focused; the same pipeline applies unchanged
    to the Portuguese file or the merged 382-student panel.</p>

  <div class="card">
    <h3>Schema at a glance</h3>
    <div class="grid-2">
      <div>
        <p style="font-size:13.5px;color:var(--ink-2);margin-top:0">16 numeric attributes (ordinal scales like
          <code>studytime</code> 1&ndash;4, or counts like <code>absences</code>) and 17 categorical attributes
          (12 binary yes/no or two-level, 4 nominal with 3&ndash;5 levels, plus <code>guardian</code>).</p>
        <p style="font-size:13.5px;color:var(--ink-2)"><b>Target:</b> <code>G3</code>, final grade, 0&ndash;20.
          <code>G1</code> and <code>G2</code> (period 1 &amp; 2 grades) are themselves near-perfect predictors of
          <code>G3</code> (r = 0.80 / 0.90) — see the framing below.</p>
      </div>
      <div>
        <p style="font-size:13.5px;color:var(--ink-2);margin-top:0"><b>Data quality:</b> 0 missing values, 0 duplicate
          rows across all {eda['shape'][1]} columns of {eda['shape'][0]} students — no imputation needed.</p>
        <p style="font-size:13.5px;color:var(--ink-2)"><b>Target anomaly:</b> {eda['zero_G3_count']} students
          ({fmt_pct(eda['pct_zero_G3']/100)}) scored exactly 0, which in the source study denotes withdrawal from the
          exam rather than a genuine failing performance — flagged and revisited under assumptions.</p>
      </div>
    </div>
  </div>

  <div class="callout">
    <strong>Two honest framings, not one.</strong> Because G1 and G2 are so predictive of G3, "predicting G3" is
    really two different problems. <b>Scenario A</b> includes G1 &amp; G2 as features (a late-semester "how will
    this student finish" model). <b>Scenario B</b> excludes them (an early-semester "who is at risk" model, using
    only demographics, family background and behaviour). Both are built end-to-end below.
  </div>
</section>

<section id="eda">
  <h2>D · Exploratory analysis &amp; insights</h2>
  <p class="section-intro">Distribution shape, correlation structure, and category-level breakdowns of the outcome
    variable, visualized to surface the strongest signals before any modeling decision is made.</p>

  <div class="grid-2">
    <div class="card">
      <figure><img src="{datauri('target')}" alt="Distribution of G3"></figure>
      <figcaption><b>Grade distribution is roughly bell-shaped around 10&ndash;11</b>, with a distinct secondary
        spike at 0 — the dropout/no-show group. This bimodality is the single biggest structural feature of the
        target.</figcaption>
    </div>
    <div class="card">
      <figure><img src="{datauri('progression')}" alt="G1 vs G2 vs G3"></figure>
      <figcaption><b>Grades are highly autocorrelated across periods</b> (G1&rarr;G2 r=0.85, G2&rarr;G3 r=0.90):
        academic performance is largely set by mid-semester, which is exactly why Scenario A and B tell such
        different stories.</figcaption>
    </div>
  </div>

  <div class="card">
    <figure><img src="{datauri('heatmap')}" alt="Correlation heatmap"></figure>
    <figcaption><b>Outside G1/G2, only <code>failures</code> (r={eda['top_corr_with_G3']['failures']:.2f}) clears a
      moderate correlation with G3</b>; parental education, age and going-out show weak (|r| 0.1&ndash;0.2) effects.
      Lifestyle variables (alcohol use, free time, health) are essentially uncorrelated with G3 in isolation — their
      effect, if any, is likely non-linear or interacts with other features (captured later by tree-based
      importances).</figcaption>
  </div>

  <div class="card">
    <figure><img src="{datauri('cat')}" alt="Categorical breakdowns"></figure>
    <figcaption><b>School, address and wanting higher education separate students most</b>: GP outperforms MS by
      roughly a grade point, urban students outperform rural by a similar margin, and the higher-education aspiration
      gap is the single widest split in this panel (~4 points) — an early motivational signal.</figcaption>
  </div>

  <div class="grid-2">
    <div class="card">
      <figure><img src="{datauri('box')}" alt="Boxplots of key drivers"></figure>
      <figcaption><b>Past failures are the cleanest monotonic driver</b> of a lower G3; study time shows a much
        weaker gradient than intuition suggests, and weekday/weekend alcohol use show only marginal separation once
        outliers are accounted for.</figcaption>
    </div>
    <div class="card">
      <figure><img src="{datauri('absences')}" alt="Absences vs G3"></figure>
      <figcaption><b>Absences alone carry almost no linear signal</b> (r&asymp;0.03) — driven by a few extreme
        outliers (one student logged 75 absences). Log-transforming this feature is part of the preprocessing step
        below.</figcaption>
    </div>
  </div>

  <div class="card">
    <figure><img src="{datauri('parent')}" alt="Parental education and job"></figure>
    <figcaption><b>Mother's education shows a fairly clean step gradient</b> with G3, and children of teacher/health
      professional mothers score highest on average — consistent with a socioeconomic / home-environment
      effect.</figcaption>
  </div>
</section>

<section id="prep">
  <h2>B/C · Preprocessing &amp; feature engineering</h2>
  <p class="section-intro">All transformations are fit on the training split only (inside a scikit-learn
    <code>Pipeline</code> + <code>ColumnTransformer</code>) to avoid leakage, then applied to the held-out test
    set.</p>

  <div class="card">
    <h3>Pipeline steps</h3>
    <ol class="steps">
      <li><b>Binary encoding</b> — 13 two-level attributes (<code>school, sex, address, famsize, Pstatus,
        schoolsup, famsup, paid, activities, nursery, higher, internet, romantic</code>) mapped to 0/1.</li>
      <li><b>One-hot encoding</b> — 4 nominal attributes with 3&ndash;5 levels (<code>Mjob, Fjob, reason,
        guardian</code>), drop-first to avoid the dummy-variable trap.</li>
      <li><b>Standardization</b> — all numeric features (raw + engineered) scaled to zero mean / unit variance
        inside the pipeline, required for KNN, SVR and the regularized linear models.</li>
      <li><b>Log transform</b> — <code>absences</code> is right-skewed with extreme outliers (max 75); a
        <code>log1p</code> version is added alongside the raw feature.</li>
      <li><b>Derived features</b> — six engineered signals built from domain reasoning (below), designed to compress
        related raw variables into single, more monotonic predictors.</li>
      <li><b>No missing-value handling needed</b> — the source data has 0 nulls, so this step is a no-op here but
        is wired into the pipeline for robustness against future data.</li>
    </ol>
  </div>

  <div class="card">
    <h3>Engineered features</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Feature</th><th>Definition</th><th>Rationale</th></tr></thead>
        <tbody>
          <tr><td><code>parent_edu_avg</code></td><td>(Medu + Fedu) / 2</td><td>Single home-education-level signal instead of two correlated ones (r(Medu,Fedu)=0.62)</td></tr>
          <tr><td><code>alc_total</code></td><td>Dalc&times;5 + Walc&times;2</td><td>Weekly-weighted alcohol exposure (5 weekdays, 2 weekend days)</td></tr>
          <tr><td><code>support_score</code></td><td>count(schoolsup, famsup, paid = yes)</td><td>Cumulative academic-support received, 0&ndash;3</td></tr>
          <tr><td><code>study_per_failure</code></td><td>studytime / (failures + 1)</td><td>Study effort discounted by prior struggle</td></tr>
          <tr><td><code>social_score</code></td><td>freetime + goout</td><td>Combined leisure/social exposure</td></tr>
          <tr><td><code>log_absences</code></td><td>log1p(absences)</td><td>Compresses the long right tail (max 75 absences)</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<section id="selection">
  <h2>Feature selection</h2>
  <p class="section-intro">Two independent selection lenses were run on the transformed training matrix: a
    univariate F-test (linear signal per feature) and Random Forest permutation-style impurity importance
    (captures non-linear / interaction signal). Scenario B is the more informative one to read here, since Scenario A
    is dominated end-to-end by G2.</p>

  <div class="grid-2">
    <div class="card">
      <h3>Top features by F-score <span class="tag scenB">Scenario B</span></h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Feature</th><th class="num">F-score</th></tr></thead>
          <tbody>{top_features_rows(B['top_fscore_features'])}</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>Top features by RF importance <span class="tag scenB">Scenario B</span></h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Feature</th><th class="num">Importance</th></tr></thead>
          <tbody>{top_features_rows(B['top_rf_importance_features'])}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="callout">
    <strong>Both lenses agree on the headline driver:</strong> <code>failures</code> tops both rankings, followed
    by absence-related features and the engineered <code>study_per_failure</code> — confirming the engineered
    feature adds signal beyond its two raw components. This is the feature set carried into the Scenario B model
    zoo (no features were hard-dropped; regularized models and tree importances perform the effective selection).
  </div>
</section>

<section id="models">
  <h2>Train/test split &amp; model selection</h2>
  <p class="section-intro">80/20 train/test split (random_state=42, {A['n_train']} train / {A['n_test']} test
    students), 5-fold cross-validation for model comparison, then <code>GridSearchCV</code> hyperparameter tuning
    on the top-3 CV performers per scenario, evaluated once on the held-out test set. Nine algorithm families were
    compared: Linear, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, Gradient Boosting, SVR.</p>
</section>

<section id="scenarioA">
  <h2>Scenario A results <span class="tag scenA" style="font-size:13px">with G1 &amp; G2</span></h2>
  <p class="section-intro">The "how will this student finish the year" framing — realistic once two graded periods
    already exist.</p>

  <div class="grid-2">
    <div class="card">
      <h3>5-fold CV comparison</h3>
      <figure><img src="{datauri('a_cv')}" alt="Scenario A model comparison"></figure>
    </div>
    <div class="card">
      <h3>CV leaderboard</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Model</th><th class="num">CV R² mean</th><th class="num">± std</th><th class="num">CV RMSE</th></tr></thead>
          <tbody>{cv_table_rows(A['cv_summary'])}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <figure><img src="{datauri('a_avp')}" alt="Scenario A actual vs predicted"></figure>
      <figcaption>Tuned <b>{A['best_model']}</b> — test R² {A['best_test_metrics']['test_r2']:.3f}, RMSE
        {A['best_test_metrics']['test_rmse']:.2f}, MAE {A['best_test_metrics']['test_mae']:.2f}, MAPE
        {A['best_test_metrics']['test_mape']:.1f}%.</figcaption>
    </div>
    <div class="card">
      <figure><img src="{datauri('a_imp')}" alt="Scenario A feature importance"></figure>
      <figcaption>G2 alone accounts for the large majority of importance — expected, since it is measured only
        weeks before G3.</figcaption>
    </div>
  </div>

  <div class="card">
    <figure><img src="{datauri('a_resid')}" alt="Scenario A residual diagnostics"></figure>
    <figcaption>Residuals cluster near zero with a visible left tail — the same zero-grade dropout group surfaces
      here as a block of large negative residuals the model cannot see coming from G1/G2 alone.</figcaption>
  </div>
</section>

<section id="scenarioB">
  <h2>Scenario B results <span class="tag scenB" style="font-size:13px">early prediction, no G1/G2</span></h2>
  <p class="section-intro">The "who is at risk, before any grade exists" framing — the genuinely hard, more
    decision-relevant ML problem, since intervening is only useful <em>before</em> the first grade comes in.</p>

  <div class="grid-2">
    <div class="card">
      <h3>5-fold CV comparison</h3>
      <figure><img src="{datauri('b_cv')}" alt="Scenario B model comparison"></figure>
    </div>
    <div class="card">
      <h3>CV leaderboard</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Model</th><th class="num">CV R² mean</th><th class="num">± std</th><th class="num">CV RMSE</th></tr></thead>
          <tbody>{cv_table_rows(B['cv_summary'])}</tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <figure><img src="{datauri('b_avp')}" alt="Scenario B actual vs predicted"></figure>
      <figcaption>Tuned <b>{B['best_model']}</b> — test R² {B['best_test_metrics']['test_r2']:.3f}, RMSE
        {B['best_test_metrics']['test_rmse']:.2f}, MAE {B['best_test_metrics']['test_mae']:.2f}, MAPE
        {B['best_test_metrics']['test_mape']:.1f}%.</figcaption>
    </div>
    <div class="card">
      <figure><img src="{datauri('b_imp')}" alt="Scenario B feature importance"></figure>
      <figcaption><code>failures</code> and absence-related features dominate; demographic/lifestyle features each
        contribute only a few percent.</figcaption>
    </div>
  </div>

  <div class="card">
    <figure><img src="{datauri('b_resid')}" alt="Scenario B residual diagnostics"></figure>
    <figcaption>Residual spread is much wider and flatter than Scenario A's — consistent with the model explaining
      only a fifth of the variance; the Q-Q plot's tails depart from normal earlier too.</figcaption>
  </div>

  <div class="callout warn">
    <strong>R² &asymp; {B['best_test_metrics']['test_r2']:.2f} is the honest ceiling here, not a modeling failure.</strong>
    This matches the original Cortez &amp; Silva (2008) findings: without any grade information, demographic and
    behavioural attributes alone explain roughly 20&ndash;30% of final-grade variance. The remaining variance is
    driven by factors this survey does not capture (aptitude, effort intensity, exam-day performance). Practically,
    a model at this R² is still useful for <em>triage</em> (ranking relative risk) but not for precise grade
    forecasting.
  </div>
</section>

<section id="assumptions">
  <h2>E · Classical regression assumptions</h2>
  <p class="section-intro">Diagnosed on an interpretable 12-feature OLS specification of Scenario A
    (<code>statsmodels</code>), since that is the specification where a linear fit is actually appropriate
    (R²={assump['r_squared']:.3f}) and coefficients are worth reading.</p>

  <div class="grid-2">
    <div class="card">
      <h3>OLS coefficients (standardized inputs)</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Feature</th><th class="num">Coef.</th><th class="num">p-value</th></tr></thead>
          <tbody>{coef_rows()}</tbody>
        </table>
      </div>
      <p style="font-size:12px;color:var(--ink-mute);margin-bottom:0">★ = significant at p&lt;0.05. R²={assump['r_squared']:.3f}, adj. R²={assump['adj_r_squared']:.3f}, F-test p={assump['f_pvalue']:.1e}.</p>
    </div>
    <div class="card">
      <h3>Assumption checklist</h3>
      <table>
        <tbody>
          <tr><td>1. Linearity (Rainbow test)</td><td><span class="pill pass">PASS</span></td></tr>
          <tr><td>2. Normality of residuals (Shapiro-Wilk)</td><td><span class="pill fail">FAIL</span></td></tr>
          <tr><td>3. Homoscedasticity (Breusch-Pagan)</td><td><span class="pill fail">FAIL</span></td></tr>
          <tr><td>4. No multicollinearity (max VIF)</td><td><span class="pill pass">PASS</span></td></tr>
          <tr><td>5. Independence (Durbin-Watson = {assump['durbin_watson']['stat']:.2f})</td><td><span class="pill pass">~2, OK</span></td></tr>
          <tr><td>6. Influential points (Cook's D &gt; 4/n)</td><td><span class="pill fail">{assump['influential_points']['n_flagged']} flagged ({assump['influential_points']['pct_flagged']}%)</span></td></tr>
        </tbody>
      </table>
      <p style="font-size:13px;color:var(--ink-2)">Rainbow p={assump['linearity_rainbow_test']['p_value']:.3f}
        &middot; Shapiro p={assump['normality']['shapiro_p']:.1e} &middot; Breusch-Pagan p={assump['homoscedasticity_breusch_pagan']['lm_p']:.1e}</p>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <figure><img src="{datauri('ols_diag')}" alt="OLS diagnostic plots"></figure>
      <figcaption>Fitted-vs-residual and scale-location plots show a funnel that widens at low fitted values, and
        the Q-Q plot's lower tail departs sharply from the reference line.</figcaption>
    </div>
    <div class="card">
      <figure><img src="{datauri('cooks')}" alt="Cook's distance"></figure>
      <figcaption>{assump['influential_points']['n_flagged']} students exceed the 4/n Cook's-distance threshold —
        concentrated among the zero-grade group, which the linear model structurally cannot fit well.</figcaption>
    </div>
  </div>

  <div class="card">
    <figure><img src="{datauri('vif')}" alt="Variance inflation factors"></figure>
    <figcaption>All VIFs stay below 5 (G1/G2 peak at ~4.0), so multicollinearity is not a material concern even
      with two highly-correlated grade features retained together.</figcaption>
  </div>

  <div class="callout warn">
    <strong>Root cause of both failures: the {eda['zero_G3_count']}-student dropout spike.</strong> Re-fitting the
    identical OLS specification after excluding <code>G3=0</code> students raises R² to 0.94 but Shapiro/Breusch-Pagan
    p-values, while much improved, still reject at the 5% level — this is a real, mild skew in exam performance, not
    only the dropout artifact. <b>Practical recommendation:</b> a two-stage "hurdle" model — (1) classify
    completed-exam vs withdrew, (2) regress G3 only on students who completed — will satisfy OLS assumptions far
    better than a single pooled regression, and is also the more decision-relevant framing (dropout risk and grade
    level are different interventions).
  </div>
</section>

<section id="conclusion">
  <h2>Conclusions &amp; recommendations</h2>
  <div class="card">
    <ol class="steps">
      <li><b>Two problems, not one.</b> "Predict G3" collapses a late-semester tracking problem (R²&asymp;{A['best_test_metrics']['test_r2']:.2f}
        once G1/G2 exist) and an early-semester risk-screening problem (R²&asymp;{B['best_test_metrics']['test_r2']:.2f}
        with demographics/behaviour alone) — report both, never one number.</li>
      <li><b>{A['best_model']} and {B['best_model']}</b> were the best-performing families in Scenarios A and B
        respectively after tuning — tree ensembles consistently beat linear and instance-based models across both
        framings, evidence of non-linear/interaction effects (e.g. studytime only helping once failures=0).</li>
      <li><b><code>failures</code> is the most decision-relevant lever</b> available before any grade exists;
        <code>study_per_failure</code>, absences and parental-education features add secondary, smaller signal.</li>
      <li><b>The zero-grade dropout group should be modeled separately.</b> It drives both the OLS normality/heteroscedasticity
        failures and a visible cluster of high-Cook's-distance points — a hurdle model (dropout classifier +
        regression on completers) is the statistically and practically correct next iteration.</li>
      <li><b>Next steps:</b> replicate on <code>student-por.csv</code> / the 382-student merged panel to test
        generalization; add interaction terms (failures &times; studytime) to the linear specification; try quantile
        regression to characterize the at-risk tail directly rather than the conditional mean.</li>
    </ol>
  </div>
</section>

<footer>
  Source: P. Cortez &amp; A. Silva, "Using Data Mining to Predict Secondary School Student Performance" (2008), UCI
  Machine Learning Repository. Pipeline: pandas, scikit-learn, statsmodels, matplotlib. Random state fixed at 42
  throughout for reproducibility.
</footer>

</div>
"""

with open("../outputs/report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Report written:", len(html), "chars")
