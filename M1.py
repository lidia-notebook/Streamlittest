# app.py
# -----------------------
# Customer Segmentation Explorer (K-Means, PCA)
# Works with:
#  - segmented_customers.csv  (must include a `cluster` column)
#  - pca_coordinates.csv      (pc1, pc2, cluster)
#  - kmeans_model.joblib      (optional, for metadata)
#  - preprocessor_pipeline.joblib (optional, for future scoring)
#
# Run:  streamlit run app.py
# -----------------------

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Segmentation Explorer",
    page_icon="📊",
    layout="wide"
)

# light blue vibe (extra polish beyond config.toml)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #E9F4FF 0%, #F6FBFF 100%) !important;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }
    .metric-card {
        background: #FFFFFFcc;
        border: 1px solid #D7ECFF;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(15, 76, 129, 0.05);
    }
    .small-muted { color:#4b6983; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer: str | io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)

@st.cache_data(show_spinner=False)
def load_local_if_exists(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def get_cluster_sizes(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df["cluster"].value_counts(dropna=False).sort_index()
    pct = (sizes / sizes.sum() * 100).round(2)
    out = pd.DataFrame({"cluster": sizes.index, "count": sizes.values, "percent": pct.values})
    return out

def numeric_columns(df: pd.DataFrame) -> list[str]:
    # exclude the cluster label and any obvious non-numeric IDs
    cand = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cand if c.lower() not in {"cluster", "id", "customer_id"}]

def categorical_columns(df: pd.DataFrame) -> list[str]:
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [c for c in cats if c.lower() not in {"id", "customer_id"}]

def pct_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby("cluster")[col]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent")
        .round(1)
        .unstack(fill_value=0)
        .sort_index()
    )

def plot_cluster_sizes(df_sizes: pd.DataFrame):
    fig = px.bar(
        df_sizes, x="cluster", y="count",
        text="percent",
        title="Cluster Size (count) with %",
        labels={"cluster": "Cluster", "count": "Count"},
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title="Count", xaxis_title="Cluster", bargap=0.25)
    st.plotly_chart(fig, use_container_width=True)

def plot_pca_scatter(df_pca: pd.DataFrame):
    fig = px.scatter(
        df_pca, x="pc1", y="pc2", color="cluster",
        color_discrete_sequence=px.colors.qualitative.Set3,
        opacity=0.75,
        title="PCA Scatter — Cluster Separation"
    )
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", legend_title="Cluster")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Sidebar: Data Sources ----------
st.sidebar.header("⚙️ Data Source")

mode = st.sidebar.radio(
    "Choose source",
    ["Use local CSV files", "Upload CSVs"],
    index=0
)

segmented_df: pd.DataFrame | None = None
pca_df: pd.DataFrame | None = None

if mode == "Use local CSV files":
    segmented_df = load_local_if_exists("segmented_customers.csv")
    pca_df = load_local_if_exists("pca_coordinates.csv")

    if segmented_df is None:
        st.sidebar.warning("`segmented_customers.csv` not found. Switch to **Upload CSVs** or place file in app directory.")
    if pca_df is None:
        st.sidebar.info("`pca_coordinates.csv` not found. PCA tab will be hidden until provided.")
else:
    up_seg = st.sidebar.file_uploader("Upload segmented_customers.csv", type=["csv"])
    up_pca = st.sidebar.file_uploader("Upload pca_coordinates.csv (optional)", type=["csv"])
    if up_seg:
        segmented_df = load_csv(up_seg)
    if up_pca:
        pca_df = load_csv(up_pca)

# Optional artifacts
with st.sidebar.expander("Optional artifacts"):
    up_model = st.file_uploader("kmeans_model.joblib (optional)", type=["joblib", "pkl"])
    up_pre = st.file_uploader("preprocessor_pipeline.joblib (optional)", type=["joblib", "pkl"])
    if up_model:
        try:
            model = joblib.load(up_model)
            st.sidebar.success("Loaded K-Means model.")
        except Exception as e:
            st.sidebar.error(f"Model load error: {e}")
    if up_pre:
        try:
            preproc = joblib.load(up_pre)
            st.sidebar.success("Loaded preprocessor pipeline.")
        except Exception as e:
            st.sidebar.error(f"Preprocessor load error: {e}")

# ---------- Guard ----------
if segmented_df is None:
    st.stop()

# ---------- Header ----------
st.title("📊 Customer Segmentation Explorer")
st.caption("Light-blue Streamlit UI — K-Means clusters with persona insights and PCA visualization")

# ---------- Top Metrics ----------
sizes = get_cluster_sizes(segmented_df)
n_clusters = sizes.shape[0]
n_rows = segmented_df.shape[0]
largest = sizes.loc[sizes["count"].idxmax()]

colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Customers", f"{n_rows:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with colB:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Number of Clusters", n_clusters)
    st.markdown('</div>', unsafe_allow_html=True)
with colC:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Largest Cluster", f"{int(largest['cluster'])}")
    st.markdown('<p class="small-muted">Share: {:.2f}%</p>'.format(largest["percent"]), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with colD:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if "silhouette" in segmented_df.columns:
        st.metric("Mean Silhouette", f"{segmented_df['silhouette'].mean():.3f}")
    else:
        st.metric("Mean Silhouette", "—")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------- Tabs ----------
tabs = st.tabs(["Overview", "Cluster Explorer", "PCA Scatter", "Downloads"])

# --- Overview
with tabs[0]:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Cluster Size Distribution")
        plot_cluster_sizes(sizes)
    with right:
        st.subheader("Cluster Share Table")
        st.dataframe(sizes, use_container_width=True)

# --- Cluster Explorer
with tabs[1]:
    st.subheader("Cluster Explorer")
    chosen = st.selectbox("Choose a cluster", options=sorted(segmented_df["cluster"].unique()))
    dfc = segmented_df[segmented_df["cluster"] == chosen].copy()

    ncols = numeric_columns(segmented_df)
    ccols = categorical_columns(segmented_df)

    # Numeric summary
    exp_num = st.expander("📈 Numeric Feature Summary (mean vs overall)", expanded=True)
    with exp_num:
        if ncols:
            cluster_means = dfc[ncols].mean().to_frame("cluster_mean")
            overall_means = segmented_df[ncols].mean().to_frame("overall_mean")
            comp = cluster_means.join(overall_means)
            comp["diff"] = comp["cluster_mean"] - comp["overall_mean"]
            st.dataframe(comp.round(3), use_container_width=True)

            topdiff = comp["diff"].abs().sort_values(ascending=False).head(10).index.tolist()
            fig = px.bar(
                comp.loc[topdiff, :].reset_index(),
                x="index", y="diff",
                title="Top Deviations from Overall Mean (abs)",
                labels={"index": "Feature", "diff": "Cluster Mean - Overall Mean"},
                color="diff",
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns detected.")

    # Categorical summary
    exp_cat = st.expander("🧩 Categorical Distribution (percent inside cluster)", expanded=False)
    with exp_cat:
        if ccols:
            cat_col = st.selectbox("Pick a categorical column", options=ccols)
            table = dfc[cat_col].value_counts(normalize=True).mul(100).round(1).rename("percent").to_frame()
            st.dataframe(table, use_container_width=True)
            fig = px.bar(
                table.reset_index(), x="index", y="percent",
                labels={"index": cat_col, "percent": "% in cluster"},
                color="percent",
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns detected.")

# --- PCA Scatter
with tabs[2]:
    st.subheader("PCA Scatter")
    if pca_df is None:
        st.info("Upload or place `pca_coordinates.csv` to enable this plot.")
    else:
        plot_pca_scatter(pca_df)

# --- Downloads
with tabs[3]:
    st.subheader("Downloads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download segmented_customers.csv",
            data=segmented_df.to_csv(index=False).encode("utf-8"),
            file_name="segmented_customers.csv",
            mime="text/csv"
        )
    with c2:
        if pca_df is not None:
            st.download_button(
                "Download pca_coordinates.csv",
                data=pca_df.to_csv(index=False).encode("utf-8"),
                file_name="pca_coordinates.csv",
                mime="text/csv"
            )
        else:
            st.button("pca_coordinates.csv (not available)", disabled=True)
    with c3:
        meta = {
            "n_rows": int(n_rows),
            "n_clusters": int(n_clusters),
            "clusters": sizes["cluster"].tolist(),
            "cluster_counts": sizes["count"].tolist(),
            "cluster_percents": sizes["percent"].tolist(),
        }
        st.download_button(
            "Download project_meta.json",
            data=json.dumps(meta, indent=2).encode("utf-8"),
            file_name="project_meta.json",
            mime="application/json"
        )

st.caption("© Segmentation Explorer — light-blue theme")
