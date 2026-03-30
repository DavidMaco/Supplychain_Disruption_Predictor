"""
Supply Chain Disruption Predictor - Interactive Dashboard
Streamlit multi-tab dashboard covering model performance, supplier risk,
pending-order predictions, feature importance, and lead-time analysis.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Disruption Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional: plotly for charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

_ROOT = os.path.dirname(os.path.abspath(__file__))


# ===================================================================
# Helpers: safe file loaders
# ===================================================================

def _load_json(filename: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(_ROOT, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_csv(filename: str) -> Optional[pd.DataFrame]:
    path = os.path.join(_ROOT, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _missing(label: str) -> None:
    st.warning(
        f"**{label}** not found. Run the pipeline first:\n\n"
        "```\npython run_next_phase.py\n```"
    )


# ===================================================================
# Tab 1: Overview KPIs
# ===================================================================

def tab_overview() -> None:
    st.header("Pipeline Overview")

    metrics = _load_json("model_metrics.json")
    biz = _load_json("business_impact.json")
    cmp = _load_json("model_comparison.json")
    lt = _load_json("lead_time_summary.json")

    if not metrics and not biz:
        _missing("model_metrics.json and business_impact.json")
        return

    # --- KPI row 1: Model performance ---
    st.subheader("Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)

    if metrics:
        c1.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%")
        c2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        c3.metric("Recall (Late)", f"{metrics.get('recall_late', 0)*100:.1f}%")
        c4.metric("F1 Score", f"{metrics.get('f1_late', 0):.4f}")
        thr = metrics.get("threshold_used", 0.5)
        c5.metric("Decision Threshold", f"{thr:.3f}", help=f"Strategy: {metrics.get('threshold_strategy','')}")

    # CV AUC row
    if metrics:
        cv_mean = metrics.get("cv_roc_auc_mean", 0)
        cv_std = metrics.get("cv_roc_auc_std", 0)
        split = metrics.get("split_method", "unknown")
        smote = "Yes" if metrics.get("smote_applied") else "No"
        st.caption(
            f"CV ROC-AUC: **{cv_mean:.4f} ± {cv_std:.4f}** | "
            f"Split method: **{split}** | SMOTE: **{smote}**"
        )

    st.divider()

    # --- KPI row 2: Business impact ---
    st.subheader("Business Impact (Scenario-Based)")
    if biz:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("High-Risk Orders", biz.get("high_risk_orders", "—"))
        hr_val = biz.get("high_risk_value_ngn", 0)
        b2.metric("High-Risk Value", f"₦{hr_val:,.0f}")
        savings = biz.get("potential_savings_ngn", 0)
        b3.metric("Scenario Savings", f"₦{savings:,.0f}")
        conf = biz.get("avg_prediction_confidence", 0)
        b4.metric("Avg Confidence", f"{conf*100:.1f}%")
        st.caption(f"⚠️ {biz.get('_disclaimer', '')}")

    st.divider()

    # --- KPI row 3: Lead times ---
    st.subheader("Lead-Time Health")
    if lt:
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("On-Time Rate", f"{lt.get('on_time_rate_pct', 0):.1f}%")
        l2.metric("Avg Variance", f"{lt.get('avg_variance_days', 0):.1f} days")
        l3.metric("Avg Delay (When Late)", f"{lt.get('avg_delay_when_late_days', 0):.1f} days")
        l4.metric("High-EW Suppliers", len(lt.get("high_early_warning_suppliers", [])))
    else:
        _missing("lead_time_summary.json")

    st.divider()

    # --- Best model from comparison ---
    if cmp:
        st.subheader("Model Comparison Winner")
        st.success(
            f"**{cmp.get('best_model', '—')}** achieved the highest CV ROC-AUC: "
            f"**{cmp.get('best_cv_roc_auc', 0):.4f}**"
        )


# ===================================================================
# Tab 2: Model Comparison
# ===================================================================

def tab_model_comparison() -> None:
    st.header("Model Comparison: RandomForest vs XGBoost")

    cmp = _load_json("model_comparison.json")
    if not cmp:
        _missing("model_comparison.json")
        return

    results = cmp.get("results", [])
    if not results:
        st.info("No comparison results available.")
        return

    display_cols = [
        "model_name", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std",
        "recall_late", "precision_late", "f1_late", "test_accuracy", "threshold_used",
    ]
    df = pd.DataFrame(results)[[c for c in display_cols if c in pd.DataFrame(results).columns]]
    df.columns = [
        c.replace("_", " ").title()
        for c in df.columns
    ]

    best = cmp.get("best_model", "")
    st.dataframe(
        df.style.highlight_max(
            subset=[c for c in df.columns if c not in ("Model Name", "Threshold Used")],
            color="#d4edda",
        ),
        use_container_width=True,
    )
    st.caption(
        f"Split: **{cmp.get('split_method', '—')}**  |  "
        f"SMOTE: **{'Yes' if cmp.get('smote_applied') else 'No'}**  |  "
        f"Winner: **{best}**"
    )

    if _PLOTLY and results:
        fig = go.Figure()
        for r in results:
            fig.add_trace(go.Bar(
                name=r["model_name"],
                x=["ROC-AUC", "CV-AUC", "Recall", "F1", "Accuracy"],
                y=[r["roc_auc"], r["cv_roc_auc_mean"], r["recall_late"],
                   r["f1_late"], r["test_accuracy"]],
            ))
        fig.update_layout(
            barmode="group",
            title="Model Metrics Comparison",
            yaxis={"range": [0, 1]},
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 3: Supplier Risk Scorecard
# ===================================================================

def tab_supplier_risk() -> None:
    st.header("Supplier Risk Scorecard")

    df = _load_csv("supplier_risk_scorecard.csv")
    if df is None:
        _missing("supplier_risk_scorecard.csv")
        return

    # Colour map for risk scores
    def _colour_risk(val: float) -> str:
        if val >= 60:
            return "background-color: #f8d7da"
        if val >= 30:
            return "background-color: #fff3cd"
        return "background-color: #d4edda"

    sort_col = st.selectbox("Sort by", df.columns.tolist(), index=df.columns.tolist().index("current_risk_score")
                            if "current_risk_score" in df.columns else 0)
    ascending = st.checkbox("Ascending", value=False)

    display_df = df.sort_values(sort_col, ascending=ascending)
    st.dataframe(
        display_df.style.applymap(
            _colour_risk, subset=["current_risk_score"] if "current_risk_score" in df.columns else []
        ),
        use_container_width=True,
        height=500,
    )

    if _PLOTLY and "current_risk_score" in df.columns and "supplier_name" in df.columns:
        fig = px.bar(
            df.sort_values("current_risk_score", ascending=True),
            x="current_risk_score",
            y="supplier_name",
            orientation="h",
            color="current_risk_score",
            color_continuous_scale=["#28a745", "#ffc107", "#dc3545"],
            title="Supplier Risk Scores",
            labels={"current_risk_score": "Risk Score", "supplier_name": "Supplier"},
        )
        fig.update_layout(height=max(300, len(df) * 30))
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 4: Pending Orders
# ===================================================================

def tab_pending_orders() -> None:
    st.header("Pending Order Risk Predictions")

    df = _load_csv("predictions_pending_orders.csv")
    if df is None:
        _missing("predictions_pending_orders.csv")
        return

    # Summary counts
    if "predicted_risk_level" in df.columns:
        vc = df["predicted_risk_level"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High Risk", int(vc.get("High Risk", 0)))
        c2.metric("🟡 Medium Risk", int(vc.get("Medium Risk", 0)))
        c3.metric("🟢 Low Risk", int(vc.get("Low Risk", 0)))

    # Risk level filter
    risk_filter = st.multiselect(
        "Filter by risk level",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk"],
    )
    filtered = df[df["predicted_risk_level"].isin(risk_filter)] if "predicted_risk_level" in df.columns else df

    def _row_colour(row: pd.Series) -> list:
        level = row.get("predicted_risk_level", "")
        colour_map = {"High Risk": "#f8d7da", "Medium Risk": "#fff3cd", "Low Risk": "#d4edda"}
        bg = colour_map.get(str(level), "")
        return [f"background-color: {bg}"] * len(row)

    st.dataframe(
        filtered.sort_values("predicted_late_probability", ascending=False)
        .style.apply(_row_colour, axis=1),
        use_container_width=True,
        height=500,
    )

    if _PLOTLY and "predicted_late_probability" in df.columns:
        fig = px.histogram(
            df,
            x="predicted_late_probability",
            color="predicted_risk_level" if "predicted_risk_level" in df.columns else None,
            nbins=20,
            title="Distribution of Predicted Late Probabilities",
            labels={"predicted_late_probability": "P(Late)"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 5: Feature Importance
# ===================================================================

def tab_feature_importance() -> None:
    st.header("Feature Importance")

    df = _load_csv("feature_importance.csv")
    if df is None:
        _missing("feature_importance.csv")
        return

    if "feature" not in df.columns or "importance" not in df.columns:
        st.error("Unexpected columns in feature_importance.csv")
        return

    df_sorted = df.sort_values("importance", ascending=False)

    if _PLOTLY:
        fig = px.bar(
            df_sorted,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance (Random Forest / Best Model)",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig.update_layout(
            height=max(400, len(df_sorted) * 25),
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_sorted.set_index("feature")["importance"])

    st.dataframe(df_sorted, use_container_width=True)


# ===================================================================
# Tab 6: Lead-Time Analysis
# ===================================================================

def tab_lead_times() -> None:
    st.header("Lead-Time Analysis")

    lt_sum = _load_json("lead_time_summary.json")
    lt_detail = _load_csv("lead_time_analysis.csv")
    lt_supplier = _load_csv("lead_time_analysis_by_supplier.csv")

    if lt_sum is None and lt_detail is None:
        _missing("lead_time_analysis.csv / lead_time_summary.json")
        return

    if lt_sum:
        st.subheader("Aggregate Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Orders", lt_sum.get("total_completed_orders", "—"))
        s2.metric("On-Time Rate", f"{lt_sum.get('on_time_rate_pct', 0):.1f}%")
        s3.metric("Avg Variance", f"{lt_sum.get('avg_variance_days', 0):.1f} days")
        s4.metric("Max Delay", f"{lt_sum.get('max_delay_days', 0):.1f} days")

        st.caption(
            f"Worst week: **{lt_sum.get('worst_week_for_lateness', 'N/A')}**  |  "
            f"Best week: **{lt_sum.get('best_week_for_on_time', 'N/A')}**  |  "
            f"ℹ️ {lt_sum.get('_note', '')}"
        )

    # Supplier breakdown table
    if lt_supplier is not None:
        st.subheader("Supplier Lead-Time Breakdown")

        def _row_colour_lt(row: pd.Series) -> list:
            score = row.get("early_warning_score", 0)
            if score >= 60:
                bg = "#f8d7da"
            elif score >= 30:
                bg = "#fff3cd"
            else:
                bg = "#d4edda"
            return [f"background-color: {bg}"] * len(row)

        st.dataframe(
            lt_supplier.sort_values("early_warning_score", ascending=False)
            .style.apply(_row_colour_lt, axis=1),
            use_container_width=True,
        )

    # Distribution chart
    if _PLOTLY and lt_detail is not None and "lead_time_variance" in lt_detail.columns:
        fig = px.histogram(
            lt_detail,
            x="lead_time_variance",
            nbins=30,
            title="Distribution of Lead-Time Variance (days late / early)",
            labels={"lead_time_variance": "Variance (days)"},
            color_discrete_sequence=["#0d6efd"],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="On-time boundary")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: expected vs actual lead time
    if _PLOTLY and lt_detail is not None:
        if "expected_lead_days" in lt_detail.columns and "actual_lead_days" in lt_detail.columns:
            fig2 = px.scatter(
                lt_detail,
                x="expected_lead_days",
                y="actual_lead_days",
                color="lateness_flag" if "lateness_flag" in lt_detail.columns else None,
                title="Expected vs Actual Lead Times",
                labels={
                    "expected_lead_days": "Expected (days)",
                    "actual_lead_days": "Actual (days)",
                    "lateness_flag": "Late?",
                },
                opacity=0.6,
            )
            # Diagonal = perfect on-time line
            max_val = max(lt_detail["expected_lead_days"].max(), lt_detail["actual_lead_days"].max())
            fig2.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line={"dash": "dash", "color": "grey"},
                name="On-time line",
            ))
            st.plotly_chart(fig2, use_container_width=True)


# ===================================================================
# App layout
# ===================================================================

def main() -> None:
    st.title("Supply Chain Disruption Predictor")
    st.caption("Powered by Random Forest & XGBoost | Walk-forward validation | SMOTE | Threshold optimisation")

    tabs = st.tabs([
        "📊 Overview",
        "🏆 Model Comparison",
        "🏭 Supplier Risk",
        "📦 Pending Orders",
        "🔬 Feature Importance",
        "⏱ Lead Times",
    ])

    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_model_comparison()
    with tabs[2]:
        tab_supplier_risk()
    with tabs[3]:
        tab_pending_orders()
    with tabs[4]:
        tab_feature_importance()
    with tabs[5]:
        tab_lead_times()


if __name__ == "__main__":
    main()
