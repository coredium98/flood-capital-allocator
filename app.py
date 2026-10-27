"""Flood Risk Cost-Benefit Simulation Dashboard (Streamlit entry point)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from flood_risk.charts import (
    plot_cost_vs_prevented_damage,
    plot_net_benefit_and_roi,
    plot_precision_recall_and_volume,
)
from flood_risk.model_loader import load_knn_bundle
from flood_risk.simulation import find_optimal_threshold, simulate_thresholds

ROOT_DIR = Path(__file__).resolve().parent
STYLES_PATH = ROOT_DIR / "assets" / "styles.css"

st.set_page_config(
    page_title="Flood Risk Cost-Benefit Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if STYLES_PATH.exists():
    st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    '<h1 class="main-header">🌊 Flood Risk Cost-Benefit Simulation Dashboard</h1>',
    unsafe_allow_html=True,
)
st.markdown("### Capital Allocation & Threshold Optimization Using K-Nearest Neighbors")


@st.cache_resource
def get_cached_model_bundle() -> dict[str, Any] | None:
    """Load and cache the trained KNN classifier and evaluation split."""
    try:
        return load_knn_bundle(target_neighbors=17)
    except Exception as exc:
        st.error(f"Error loading model artifact: {exc}")
        return None


model_data = get_cached_model_bundle()

if model_data is not None:
    best_model = model_data["model"]
    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    feature_names = model_data.get("feature_names")

    with st.expander("ℹ️ Model & Evaluation Dataset Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "K-Nearest Neighbors")
        with col2:
            st.metric("Number of Neighbors (k)", model_data.get("n_neighbors", 17))
        with col3:
            st.metric("Evaluation Zones", len(y_test))

        if feature_names:
            st.write(
                f"**Features ({len(feature_names)}):** {', '.join(feature_names[:6])} "
                f"(+{len(feature_names) - 6} additional predictors)"
            )

    # Sidebar parameter controls
    st.sidebar.header("🎛️ Simulation Parameters")
    st.sidebar.markdown(
        "Adjust the economic parameters below to evaluate how unit costs and mitigation "
        "effectiveness shift the optimal probability cutoff:"
    )

    cost_per_area = st.sidebar.slider(
        "💰 Cost per Intervention Area ($M)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Capital expenditure required to deploy flood prevention measures in a single zone.",
    )

    damage_per_area = st.sidebar.slider(
        "💥 Potential Damage per Flood ($M)",
        min_value=2.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Expected economic loss when an unprotected zone experiences flooding.",
    )

    effectiveness_rate = (
        st.sidebar.slider(
            "✅ Intervention Effectiveness (%)",
            min_value=40,
            max_value=100,
            value=70,
            step=5,
            help="Proportion of flood damage mitigated when an intervention is deployed.",
        )
        / 100.0
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("📊 About This Dashboard"):
        st.markdown(
            """
        This simulator converts probabilistic flood predictions into an economic capital-allocation rule:

        - **Cost per Area**: Capital outlay per mitigated zone
        - **Damage per Flood**: Economic loss avoided per true flood event
        - **Effectiveness**: Mitigation success rate

        Across 19 probability cutoffs (0.05 to 0.95), the engine identifies the threshold that maximizes expected **Net Benefit**.
        """
        )

    with st.spinner("Running cost-benefit simulation..."):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        records = simulate_thresholds(
            y_true=[int(v) for v in y_test],
            y_proba=[float(p) for p in y_proba],
            cost_per_area=cost_per_area,
            damage_per_area=damage_per_area,
            effectiveness_rate=effectiveness_rate,
        )
        optimal_result = find_optimal_threshold(records)
        results_df = pd.DataFrame(records)

    st.markdown("---")
    st.header("📈 Simulation Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric(
                "Optimal Threshold",
                f"{optimal_result['threshold']:.2f}",
                help="Probability cutoff that maximizes net economic benefit.",
            )
    with col2:
        with st.container(border=True):
            st.metric(
                "Net Benefit",
                f"${optimal_result['net_benefit']:.2f}M",
                help="Prevented flood damage minus total intervention cost.",
            )
    with col3:
        with st.container(border=True):
            st.metric(
                "ROI",
                f"{optimal_result['roi']:.1f}%",
                help="Net benefit divided by total intervention cost.",
            )
    with col4:
        with st.container(border=True):
            st.metric(
                "Areas to Intervene",
                f"{int(optimal_result['areas_flagged'])}",
                help="Number of zones exceeding the optimal probability threshold.",
            )

    st.markdown("---")
    st.subheader("🎯 Optimal Cost-Benefit Scenario")

    fin_col, perf_col = st.columns(2)
    with fin_col:
        st.markdown("**Financial Metrics:**")
        st.write(f"- **Intervention Cost:** ${optimal_result['intervention_cost']:.2f}M")
        st.write(f"- **Prevented Damage:** ${optimal_result['prevented_damage']:.2f}M")
        st.write(f"- **Net Benefit:** ${optimal_result['net_benefit']:.2f}M")
        st.write(f"- **ROI:** {optimal_result['roi']:.1f}%")

    with perf_col:
        st.markdown("**Classification Metrics at Optimal Cutoff:**")
        st.write(f"- **Precision:** {optimal_result['precision']:.1%}")
        st.write(f"- **Recall:** {optimal_result['recall']:.1%}")
        st.write(f"- **True Positives:** {int(optimal_result['true_positives'])} floods mitigated")
        st.write(f"- **Areas Flagged:** {int(optimal_result['areas_flagged'])} total interventions")

    st.markdown("---")
    st.header("📊 Visual Analysis")

    tab1, tab2, tab3 = st.tabs(
        ["💰 Cost-Benefit Analysis", "🎯 Model Performance", "📋 Data Table"]
    )

    with tab1:
        fig1 = plot_net_benefit_and_roi(results_df, optimal_result["threshold"])
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = plot_cost_vs_prevented_damage(results_df, optimal_result["threshold"])
        st.pyplot(fig2)
        plt.close(fig2)

    with tab2:
        fig3 = plot_precision_recall_and_volume(results_df, optimal_result["threshold"])
        st.pyplot(fig3)
        plt.close(fig3)

    with tab3:
        st.subheader("📋 Detailed Results Table")

        display_df = pd.DataFrame(
            {
                "Threshold": results_df["threshold"].map("{:.2f}".format),
                "Precision": results_df["precision"].map("{:.1%}".format),
                "Recall": results_df["recall"].map("{:.1%}".format),
                "Areas Flagged": results_df["areas_flagged"].astype(int),
                "True Positives": results_df["true_positives"].astype(int),
                "Intervention Cost": results_df["intervention_cost"].map("${:.2f}M".format),
                "Prevented Damage": results_df["prevented_damage"].map("${:.2f}M".format),
                "Net Benefit": results_df["net_benefit"].map("${:.2f}M".format),
                "ROI": results_df["roi"].map("{:.1f}%".format),
            }
        )

        st.dataframe(display_df, use_container_width=True, height=400)

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="cost_benefit_simulation_results.csv",
            mime="text/csv",
        )

    st.markdown("---")
    with st.expander("💡 How to Interpret These Results"):
        st.markdown(
            """
        ### Decision Framework

        - **Optimal Threshold**: Probability cutoff above which expected prevented damage exceeds marginal intervention cost.
        - **Net Benefit**: Total prevented flood damage minus total capital deployed.
        - **ROI**: Percentage return per dollar of preventive infrastructure spending.
        - **Precision**: Share of funded interventions deployed in zones that genuinely flood.
        - **Recall**: Share of actual flood zones protected by the intervention budget.
        """
        )
else:
    st.warning("⚠️ Model artifact not found in `models/knn_model.pkl`.")

st.markdown("---")
st.markdown(
    """
<div class="footer">
    <p>🌊 Flood Risk Cost-Benefit Simulation Dashboard | Built with Streamlit & scikit-learn</p>
    <p>Author: Zacharie Houle, CPA auditeur</p>
</div>
""",
    unsafe_allow_html=True,
)
