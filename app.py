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

STATE = {
    "model": None,
    "model_features": [],
    "dataset_path": None,
    "df": None,
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
    X = pd.DataFrame()
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

def valid_target_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if TARGET not in df.columns:
        return None
    y = df[TARGET]
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
    return df

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
            flash("Model loaded ✅", "success")
        except Exception as e:
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
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = os.path.join(UPLOAD_FOLDER, f"dataset_{stamp}.csv")
            df.to_csv(saved_path, index=False)
            STATE["dataset_path"] = saved_path
            STATE["df"] = df
            flash("Dataset uploaded ✅", "success")
        except Exception as e:
            flash(f"Failed to read dataset: {e}", "danger")
        return redirect(url_for("index"))

    # Batch scoring
    batch_metrics, batch_figs = {}, {}
    classification_txt, preview_scored_head_html = None, None

    if action == "score_batch" and STATE["model"] is not None and STATE["df"] is not None:
        model = STATE["model"]
        df = STATE["df"].copy()
        expected_feats = get_expected_features(model) or DEFAULT_FEATURES

        X = prepare_batch_X(df, expected_feats, use_log1p=STATE["use_log1p"])
        X = align_to_model_features(model, X)

        ncoef = getattr(model, "coef_", None)
        if ncoef is not None and X.shape[1] != ncoef.shape[1]:
            exp = get_expected_features(model)
            details = "Expected names: " + ", ".join(exp) if exp else "Check your feature list/transform."
            flash(f"Feature mismatch: model expects {ncoef.shape[1]} features but received {X.shape[1]}. {details}", "danger")
            return redirect(url_for("index"))

        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            score = getattr(model, "decision_function", None)
            if score is None:
                flash("Model has neither predict_proba nor decision_function.", "danger")
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
    if STATE["df"] is not None:
        df = STATE["df"]
        overview["df_head_html"] = df.head().to_html(classes="table table-striped table-sm", index=False)
        y = valid_target_series(df)
        if y is not None:
            bal = y.value_counts().rename({0: "Not Abandoned", 1: "Abandoned"})
            bar = px.bar(bal, title="Class Balance (target)", template="plotly_white")
            overview["target_bar_json"] = fig_to_json(bar)
        ui_feats = STATE["model_features"] or DEFAULT_FEATURES
        present = [c for c in ui_feats if c in df.columns]
        for colname in present:
            h = px.histogram(df, x=colname, nbins=30, title=colname, template="plotly_white")
            overview["feature_hist_jsons"].append(fig_to_json(h))

    return render_template(
        "index.html",
        app_name=APP_NAME,
        model_loaded=STATE["model"] is not None,
        model_features=STATE["model_features"],
        dataset_loaded=STATE["df"] is not None,
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
