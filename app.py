# app.py — S-Square Analytics (Flask, Dashboard + Recommender) — No Images
# Developed by Sanjay B and Subash V

import os
import io
import json
import pickle
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, flash
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    classification_report,
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from recommender import ProductRecommender

# -------------------------
# Config & constants
# -------------------------
APP_NAME = "S-Square Analytics — Cart Abandonment"
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_MODEL_EXT = {"pkl", "pickle", "sav", "bin"}
ALLOWED_DATA_EXT = {"csv", "xlsx", "xls"}
DEFAULT_MODEL_PATH = "train_classifier.pkl"
TARGET = "Cart_Abandoned"
DEFAULT_DASH_DATASET_CANDIDATES = [
    os.path.join("cartAbandonment", "product.csv"),  # legacy path some setups use
    "product.csv",  # legacy filename some setups use
    os.path.join("data", "product.csv"),
    os.path.join("data", "dataset.csv"),
    "dataset.csv",
]
DEFAULT_FEATURES: List[str] = [
    "No_Items_Added_InCart",
    "No_Checkout_Confirmed",
    "No_Checkout_Initiated",
    "No_Customer_Login",
    "No_Page_Viewed",
]

PRODUCTS_CSV = "products.csv"  # no images required

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "s-square-analytics"

def log(msg: str):
    # Lightweight debug logging (visible in console / dev server output)
    try:
        print(f"[dashboard] {msg}", flush=True)
    except Exception:
        pass

STATE = {
    "model": None,
    "model_features": [],
    "dataset_path": None,
    "df": None,
    "df_source": None,  # "uploaded" | "default" | None
    "last_scored_path": None,
    "threshold": 0.50,
    "use_log1p": True,
    "recommender": None,
    "products_df": None,
}

# -------------------------
# Utilities
# -------------------------
def try_load_pickle(file_like_or_path):
    if isinstance(file_like_or_path, (str, os.PathLike)):
        with open(file_like_or_path, "rb") as f:
            return pickle.load(f)
    return pickle.load(file_like_or_path)

def ensure_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype(str), errors="coerce")

def get_expected_features(model) -> List[str]:
    try:
        return list(getattr(model, "feature_names_in_", []))
    except Exception:
        return []

def prepare_batch_X(df: pd.DataFrame, feature_names: List[str], use_log1p: bool) -> pd.DataFrame:
    # Preserve row count/index even when some features are missing.
    # (If X starts empty and we assign scalars for missing cols, pandas may create a 1-row frame.)
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = ensure_numeric_series(df[col]).fillna(0)
        else:
            X[col] = 0.0
    if use_log1p:
        for col in feature_names:
            vals = np.clip(X[col].values, a_min=0, a_max=None)
            X[col] = np.log1p(vals)
    return X

def align_to_model_features(model, X: pd.DataFrame) -> pd.DataFrame:
    expected = get_expected_features(model)
    if not expected:
        return X
    for c in expected:
        if c not in X.columns:
            X[c] = 0.0
    return X[expected]

def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("__", "_")

def find_col_case_insensitive(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """Return actual column name in df matching wanted, ignoring case/whitespace/punct."""
    w = _norm_col(wanted).replace("_", "")
    for c in df.columns:
        if _norm_col(c).replace("_", "") == w:
            return c
    return None

def valid_target_series(df: pd.DataFrame) -> Optional[pd.Series]:
    target_col = TARGET if TARGET in df.columns else find_col_case_insensitive(df, TARGET)
    if not target_col:
        return None
    y = df[target_col]
    if y.dtype == "O" or str(y.dtype).startswith("category"):
        y = y.astype(str).str.strip().str.lower().map({"1": 1, "0": 0, "yes": 1, "no": 0})
    y = pd.to_numeric(y, errors="coerce")
    if y.isna().all():
        return None
    return y.astype(int)

def fig_to_json(fig) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def plot_curves(y_true: np.ndarray, p: np.ndarray):
    prec, rec, _ = precision_recall_curve(y_true, p)
    fpr, tpr, _ = roc_curve(y_true, p)

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve"))
    pr_fig.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision", template="plotly_white")

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_white")
    return pr_fig, roc_fig

def coef_bar_figure(model, features: List[str]):
    try:
        coef = np.ravel(getattr(model, "coef_", np.array([[]])))
        if coef.size == 0 or len(features) != coef.size:
            return None
        dfc = pd.DataFrame({"feature": features, "coef": coef}).sort_values("coef")
        fig = px.bar(dfc, x="coef", y="feature", orientation="h", title="Logistic Coefficients (direction only)", template="plotly_white")
        return fig
    except Exception:
        return None

def allowed_ext(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def load_dataframe(file_storage) -> pd.DataFrame:
    name = file_storage.filename.lower()
    stream = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(stream)
    else:
        df = pd.read_csv(stream, sep=None, engine="python", encoding="utf-8-sig")
    # Normalize column names early (prevents silent "no charts" due to whitespace/case mismatches)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_dataframe_from_path(path: str) -> pd.DataFrame:
    """Load a dataframe from a CSV/XLSX path on disk."""
    p = str(path)
    if p.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def get_active_dataset() -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Single source of truth for dashboard dataset.
    Priority: uploaded dataset (STATE['df'] / STATE['dataset_path']) -> default candidate file -> None
    Returns: (df, path, source)
    """
    # Uploaded dataset in memory wins
    if isinstance(STATE.get("df"), pd.DataFrame) and STATE["df"] is not None:
        return STATE["df"], STATE.get("dataset_path"), STATE.get("df_source") or "uploaded"

    # Try loading from last uploaded path if memory cleared
    up = STATE.get("dataset_path")
    if up and os.path.exists(up):
        try:
            df = load_dataframe_from_path(up)
            STATE["df"] = df
            STATE["df_source"] = "uploaded"
            return df, up, "uploaded"
        except Exception as e:
            log(f"Failed to reload uploaded dataset from {up}: {e}")

    # Fallback: default dashboard dataset (NOT products.csv)
    for cand in DEFAULT_DASH_DATASET_CANDIDATES:
        if cand and os.path.exists(cand):
            try:
                df = load_dataframe_from_path(cand)
                STATE["df"] = df
                STATE["dataset_path"] = cand
                STATE["df_source"] = "default"
                return df, cand, "default"
            except Exception as e:
                log(f"Failed to load default dataset from {cand}: {e}")

    return None, None, None

def numeric_columns_for_hist(df: pd.DataFrame, target_col: Optional[str]) -> List[str]:
    """Pick numeric-ish columns for histograms, excluding id-like and target."""
    exclude = set()
    if target_col:
        exclude.add(target_col)
    for c in df.columns:
        cn = _norm_col(c)
        if cn in {"id", "idx", "index"} or cn.endswith("id") or cn.startswith("id"):
            exclude.add(c)

    cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                cols.append(c)
            continue
        # try coercion for object columns that are numeric strings
        coerced = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
        if coerced.notna().any():
            cols.append(c)
    return cols

def ensure_default_model():
    if STATE["model"] is None and os.path.exists(DEFAULT_MODEL_PATH):
        try:
            model = try_load_pickle(DEFAULT_MODEL_PATH)
            STATE["model"] = model
            STATE["model_features"] = get_expected_features(model) or DEFAULT_FEATURES
            flash("Default model loaded ✅", "success")
        except Exception as e:
            flash(f"Could not load default model: {e}", "danger")

def ensure_recommender():
    if STATE["recommender"] is None:
        if not os.path.exists(PRODUCTS_CSV):
            flash("products.csv not found in project root. Place it next to app.py.", "warning")
            return
        try:
            rec = ProductRecommender(PRODUCTS_CSV)
            STATE["recommender"] = rec
            STATE["products_df"] = rec.df
        except Exception as e:
            flash(f"Failed to initialize recommender: {e}", "danger")

# -------------------------
# Routes: Dashboard
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    ensure_default_model()
    # Always resolve which dataset the dashboard should use
    active_df, active_path, active_src = get_active_dataset()
    if active_src:
        log(f"Active dashboard dataset: source={active_src} path={active_path} shape={getattr(active_df, 'shape', None)}")

    if "threshold" in request.form:
        try:
            STATE["threshold"] = float(request.form.get("threshold", 0.5))
        except Exception:
            STATE["threshold"] = 0.5
    if "use_log1p" in request.form:
        STATE["use_log1p"] = True
    elif request.method == "POST" and "use_log1p" not in request.form:
        STATE["use_log1p"] = False

    action = request.form.get("action")

    # Upload model
    if action == "upload_model":
        file = request.files.get("model_file")
        if not file or file.filename == "":
            flash("Please choose a model file.", "warning")
            return redirect(url_for("index"))
        if not allowed_ext(file.filename, ALLOWED_MODEL_EXT):
            flash("Unsupported model extension.", "danger")
            return redirect(url_for("index"))
        try:
            model = try_load_pickle(file.stream)
            STATE["model"] = model
            STATE["model_features"] = get_expected_features(model) or DEFAULT_FEATURES
            log(f"Model uploaded. feature_names={len(STATE['model_features'])}")
            flash("Model loaded ✅", "success")
        except Exception as e:
            log(f"Model upload failed: {e}")
            flash(f"Failed to load model: {e}", "danger")
        return redirect(url_for("index"))

    # Upload dataset
    if action == "upload_dataset":
        file = request.files.get("data_file")
        if not file or file.filename == "":
            flash("Please choose a dataset file.", "warning")
            return redirect(url_for("index"))
        if not allowed_ext(file.filename, ALLOWED_DATA_EXT):
            flash("Unsupported dataset extension.", "danger")
            return redirect(url_for("index"))
        try:
            df = load_dataframe(file)
            log(f"Dataset uploaded. shape={df.shape} cols_sample={list(df.columns[:10])}")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = os.path.join(UPLOAD_FOLDER, f"dataset_{stamp}.csv")
            df.to_csv(saved_path, index=False)
            STATE["dataset_path"] = saved_path
            STATE["df"] = df
            STATE["df_source"] = "uploaded"
            flash("Dataset uploaded ✅", "success")
        except Exception as e:
            log(f"Dataset upload failed: {e}")
            flash(f"Failed to read dataset: {e}", "danger")
        return redirect(url_for("index"))

    # Batch scoring
    batch_metrics, batch_figs = {}, {}
    classification_txt, preview_scored_head_html = None, None

    if action == "score_batch" and STATE["model"] is not None:
        model = STATE["model"]
        # Ensure we score against the same active dataset used across the dashboard
        df0, path0, src0 = get_active_dataset()
        if df0 is None:
            flash("No dataset loaded — upload a dataset first.", "warning")
            log("Batch scoring aborted: no dataset available.")
            return redirect(url_for("index"))
        df = df0.copy()
        log(f"Batch score clicked. source={src0} path={path0} df_shape={df.shape}")
        if df is None or df.empty:
            flash("Uploaded dataset has 0 rows — cannot run batch scoring. Please upload a non-empty file.", "warning")
            log("Batch scoring aborted: dataset empty.")
            return redirect(url_for("index"))

        expected_feats = get_expected_features(model) or DEFAULT_FEATURES
        if not expected_feats:
            flash("Model feature list is empty — cannot build input matrix for batch scoring.", "danger")
            log("Batch scoring aborted: expected feature list empty.")
            return redirect(url_for("index"))

        X = prepare_batch_X(df, expected_feats, use_log1p=STATE["use_log1p"])
        log(f"After prepare_batch_X. X_shape={X.shape}")
        X = align_to_model_features(model, X)
        log(f"After align_to_model_features. X_shape={X.shape}")

        # Final safety check: scikit-learn will crash on 0 rows.
        if X.shape[0] == 0:
            flash("No usable rows available for prediction after preprocessing (0 samples). Check your dataset content.", "warning")
            log("Batch scoring aborted: X has 0 rows after preprocessing/alignment.")
            return redirect(url_for("index"))

        ncoef = getattr(model, "coef_", None)
        if ncoef is not None and X.shape[1] != ncoef.shape[1]:
            exp = get_expected_features(model)
            details = "Expected names: " + ", ".join(exp) if exp else "Check your feature list/transform."
            flash(f"Feature mismatch: model expects {ncoef.shape[1]} features but received {X.shape[1]}. {details}", "danger")
            return redirect(url_for("index"))

        try:
            if X.shape[0] == 0:
                raise ValueError("0 samples after preprocessing")
            proba = model.predict_proba(X)[:, 1]
        except Exception as e:
            # Avoid hard crashes on empty / invalid matrices
            if X.shape[0] == 0:
                flash("Cannot score: 0 usable rows after preprocessing (0 samples). Please upload a non-empty dataset.", "warning")
                log(f"Batch scoring blocked in predict_proba: {e}")
                return redirect(url_for("index"))
            score = getattr(model, "decision_function", None)
            if score is None:
                flash("Model has neither predict_proba nor decision_function.", "danger")
                return redirect(url_for("index"))
            if X.shape[0] == 0:
                flash("Cannot score: 0 usable rows after preprocessing (0 samples).", "warning")
                log(f"Batch scoring blocked in decision_function: {e}")
                return redirect(url_for("index"))
            raw = score(X)
            proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

        out = df.copy()
        out["abandon_prob"] = proba
        out["abandon_pred"] = (out["abandon_prob"] >= STATE["threshold"]).astype(int)

        out_path = os.path.join(OUTPUT_FOLDER, "scored_cart_abandonment.csv")
        out.to_csv(out_path, index=False)
        STATE["last_scored_path"] = out_path
        preview_scored_head_html = out.head().to_html(classes="table table-striped table-sm", index=False)

        y = valid_target_series(df)
        if y is not None:
            try:
                batch_metrics["roc_auc"] = float(roc_auc_score(y, proba))
                batch_metrics["pr_auc"] = float(average_precision_score(y, proba))
            except Exception as e:
                flash(f"Could not compute AUCs: {e}", "warning")

            y_hat = (proba >= STATE["threshold"]).astype(int)
            cm = confusion_matrix(y, y_hat, labels=[0, 1])
            cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=["0", "1"], y=["0", "1"],
                               title="Confusion Matrix (@ threshold)", template="plotly_white")
            pr_fig, roc_fig = plot_curves(y.values, proba)

            batch_figs["cm_json"] = fig_to_json(cm_fig)
            batch_figs["pr_json"] = fig_to_json(pr_fig)
            batch_figs["roc_json"] = fig_to_json(roc_fig)

            try:
                classification_txt = classification_report(y, y_hat,
                                    target_names=["Not Abandoned", "Abandoned"], output_dict=False)
            except Exception:
                classification_txt = None
        else:
            flash(f"No numeric '{TARGET}' found — metrics/curves hidden. Add 0/1 target to evaluate.", "info")

        coef_feats = get_expected_features(model) or expected_feats
        coef_fig = coef_bar_figure(model, coef_feats)
        if coef_fig is not None:
            batch_figs["coef_json"] = fig_to_json(coef_fig)

        flash("Batch scored ✅", "success")

    # Single prediction
    single_pred = None
    if action == "single_predict" and STATE["model"] is not None:
        model = STATE["model"]
        ui_feats = STATE["model_features"] or DEFAULT_FEATURES

        values = {}
        for fname in ui_feats:
            raw = request.form.get(f"feat_{fname}", "0")
            try:
                values[fname] = float(raw)
            except Exception:
                values[fname] = 0.0

        row_df = pd.DataFrame([values])
        X1 = prepare_batch_X(row_df, ui_feats, use_log1p=STATE["use_log1p"])
        X1 = align_to_model_features(model, X1)

        ncoef = getattr(model, "coef_", None)
        if ncoef is not None and X1.shape[1] != ncoef.shape[1]:
            exp = get_expected_features(model)
            details = "Expected names: " + ", ".join(exp) if exp else "Check your feature list/transform."
            flash(f"Feature mismatch: model expects {ncoef.shape[1]} features but received {X1.shape[1]}. {details}", "danger")
        else:
            try:
                p = float(model.predict_proba(X1)[:, 1][0])
            except Exception:
                score = getattr(model, "decision_function", None)
                if score is None:
                    flash("Model has neither predict_proba nor decision_function.", "danger")
                    return redirect(url_for("index"))
                z = float(score(X1)[0])
                p = 1.0 / (1.0 + np.exp(-z))
            single_pred = {"prob": p, "decision": "Abandoned" if p >= STATE["threshold"] else "Not Abandoned"}

    # Overview figs
    overview = {"df_head_html": None, "target_bar_json": None, "feature_hist_jsons": []}
    if active_df is not None:
        df = active_df
        overview["df_head_html"] = df.head().to_html(classes="table table-striped table-sm", index=False)
        target_col = TARGET if TARGET in df.columns else find_col_case_insensitive(df, TARGET)
        y = valid_target_series(df)
        if y is not None:
            try:
                bal = y.value_counts().rename({0: "Not Abandoned", 1: "Abandoned"})
                # Build the figure with plain Python lists (avoids typed-array JSON that can render empty in some setups)
                x_labels = ["Not Abandoned", "Abandoned"]
                y_counts = [int(bal.get("Not Abandoned", 0)), int(bal.get("Abandoned", 0))]
                bar = go.Figure(data=[go.Bar(x=x_labels, y=y_counts, name="count")])
                bar.update_layout(title="Class Balance (target)", template="plotly_white", xaxis_title=target_col or TARGET, yaxis_title="count")
                overview["target_bar_json"] = fig_to_json(bar)
                log(f"Rendered class balance. target_col={target_col} counts={bal.to_dict()}")
            except Exception as e:
                log(f"Class balance render failed: {e}")
        else:
            log(f"No usable target for class balance. target_col_guess={target_col}")

        # Feature distributions: use a mix of chart types (pie/radar/scatter)
        cols = numeric_columns_for_hist(df, target_col=target_col)
        if not cols:
            log("No numeric columns found for feature charts.")
        else:
            log(f"Numeric columns selected for feature charts: {cols}")

        # 1) Pie chart for target balance (if available)
        if y is not None:
            try:
                counts = y.value_counts()
                labels = ["Not Abandoned", "Abandoned"]
                values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
                pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
                pie.update_layout(title="Target balance (pie)", template="plotly_white")
                overview["feature_hist_jsons"].append(fig_to_json(pie))
                log(f"Rendered target pie. values={values}")
            except Exception as e:
                log(f"Target pie render failed: {e}")

        # 2) Radar chart of feature means (up to 8)
        try:
            radar_cols = cols[:8]
            if radar_cols:
                means = []
                thetas = []
                for c in radar_cols:
                    s = ensure_numeric_series(df[c])
                    m = float(pd.to_numeric(s, errors='coerce').mean())
                    if np.isfinite(m):
                        means.append(m)
                        thetas.append(c)
                if means:
                    radar = go.Figure(
                        data=[go.Scatterpolar(r=means, theta=thetas, fill="toself", name="mean")]
                    )
                    radar.update_layout(
                        title="Feature profile (radar: mean)",
                        template="plotly_white",
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=False,
                    )
                    overview["feature_hist_jsons"].append(fig_to_json(radar))
                    log(f"Rendered radar. cols={thetas}")
        except Exception as e:
            log(f"Radar render failed: {e}")

        # 3) Scatter plot for two numeric features (highest variance)
        try:
            if len(cols) >= 2:
                variances = []
                for c in cols:
                    s = ensure_numeric_series(df[c])
                    v = float(pd.to_numeric(s, errors="coerce").var())
                    if np.isfinite(v):
                        variances.append((v, c))
                variances.sort(reverse=True)
                if len(variances) >= 2:
                    xcol = variances[0][1]
                    ycol = variances[1][1]
                    xs = ensure_numeric_series(df[xcol])
                    ys = ensure_numeric_series(df[ycol])
                    x_vals = pd.to_numeric(xs, errors="coerce")
                    y_vals = pd.to_numeric(ys, errors="coerce")
                    mask = x_vals.notna() & y_vals.notna()
                    x_list = x_vals[mask].astype(float).tolist()
                    y_list = y_vals[mask].astype(float).tolist()
                    if len(x_list) > 0:
                        scatter = go.Figure(data=[go.Scatter(x=x_list, y=y_list, mode="markers", marker=dict(size=5, opacity=0.6))])
                        scatter.update_layout(
                            title=f"Feature relationship (scatter): {xcol} vs {ycol}",
                            template="plotly_white",
                            xaxis_title=xcol,
                            yaxis_title=ycol,
                        )
                        overview["feature_hist_jsons"].append(fig_to_json(scatter))
                        log(f"Rendered scatter. x={xcol} y={ycol} n={len(x_list)}")
        except Exception as e:
            log(f"Scatter render failed: {e}")

        # If nothing rendered, show a demo set so the section isn't blank
        if len(overview["feature_hist_jsons"]) == 0:
            try:
                rng = np.random.default_rng(7)
                demo_vals = rng.poisson(lam=5.0, size=500).astype(int).tolist()
                demo_pie = go.Figure(data=[go.Pie(labels=["Class 0", "Class 1"], values=[300, 200], hole=0.35)])
                demo_pie.update_layout(title="Demo pie (placeholder)", template="plotly_white")

                demo_radar = go.Figure(data=[go.Scatterpolar(r=[2, 5, 3, 4, 6], theta=["F1", "F2", "F3", "F4", "F5"], fill="toself")])
                demo_radar.update_layout(title="Demo radar (placeholder)", template="plotly_white", showlegend=False)

                demo_scatter = go.Figure(data=[go.Scatter(x=demo_vals, y=rng.normal(0, 1, size=500).tolist(), mode="markers", marker=dict(size=5, opacity=0.6))])
                demo_scatter.update_layout(title="Demo scatter (placeholder)", template="plotly_white", xaxis_title="x", yaxis_title="y")

                overview["feature_hist_jsons"].extend([fig_to_json(demo_pie), fig_to_json(demo_radar), fig_to_json(demo_scatter)])
                log("Rendered demo placeholder charts (pie/radar/scatter).")
            except Exception as e:
                log(f"Demo placeholder charts failed: {e}")

    return render_template(
        "index.html",
        app_name=APP_NAME,
        model_loaded=STATE["model"] is not None,
        model_features=STATE["model_features"],
        dataset_loaded=active_df is not None,
        threshold=STATE["threshold"],
        use_log1p=STATE["use_log1p"],
        overview=overview,
        batch_metrics=batch_metrics,
        batch_figs=batch_figs,
        classification_txt=classification_txt,
        preview_scored_head_html=preview_scored_head_html,
        single_pred=single_pred,
        default_model_exists=os.path.exists(DEFAULT_MODEL_PATH),
        target_name=TARGET
    )

# -------------------------
# Downloads
# -------------------------
@app.route("/download/scored")
def download_scored():
    if not STATE["last_scored_path"] or not os.path.exists(STATE["last_scored_path"]):
        flash("No scored file available yet.", "warning")
        return redirect(url_for("index"))
    return send_file(STATE["last_scored_path"], as_attachment=True)

@app.route("/download/model")
def download_model():
    if os.path.exists(DEFAULT_MODEL_PATH):
        return send_file(DEFAULT_MODEL_PATH, as_attachment=True, download_name="train_classifier.pkl")
    flash("Default model not found.", "warning")
    return redirect(url_for("index"))

# -------------------------
# Recommender
# -------------------------
@app.route("/recs", methods=["GET", "POST"])
def recs():
    ensure_recommender()
    if STATE["recommender"] is None:
        return render_template(
            "recommender.html",
            app_name=APP_NAME,
            options=[],
            selected_title=None,
            selected=None,
            recs=[],
            default_model_exists=os.path.exists(DEFAULT_MODEL_PATH)
        )

    rec = STATE["recommender"]
    options = rec.get_all_titles()

    selected_title = None
    selected = None
    recs_df = pd.DataFrame()

    if request.method == "POST":
        selected_title = request.form.get("product_title")
        pid = rec.id_for_title(selected_title) if selected_title else None
        if pid is None:
            flash("Please pick a valid product.", "warning")
        else:
            selected = rec.df.loc[rec.df["id"] == pid].iloc[0].to_dict()
            recs_df = rec.recommend(pid, k=6)

    recs_list = recs_df.to_dict(orient="records") if not recs_df.empty else []

    return render_template(
        "recommender.html",
        app_name=APP_NAME,
        options=options,
        selected_title=selected_title,
        selected=selected,
        recs=recs_list,
        default_model_exists=os.path.exists(DEFAULT_MODEL_PATH)
    )

if __name__ == "__main__":
    app.run(debug=True)
