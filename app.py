"""
DataCleaner Pro — Streamlit Dashboard
======================================
Run with:  streamlit run app.py
"""

import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── local imports ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from cleaner import (
    build_quality_report,
    detect_anomalies_isolation_forest,
    handle_outliers,
    impute_missing,
    fix_types,
    remove_duplicates,
    suggest_cleaning_strategies,
)
from loader import load_auto, save_dataframe
from reporter import generate_html_report
from visualizer import (
    plotly_bar,
    plotly_boxplot,
    plotly_correlation_heatmap,
    plotly_histogram,
    plotly_missing_bar,
    plotly_outlier_bar,
    plotly_pie,
    plot_histogram_static,
    plot_boxplot_static,
    plot_bar_static,
    plot_correlation_heatmap_static,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataCleaner Pro",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] {background: #0f172a;}
  [data-testid="stSidebar"] {background: #1e293b;}
  h1,h2,h3 {color: #e2e8f0 !important;}
  .metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
  }
  .metric-val  {font-size: 2rem; font-weight: 700; color: #22d3ee;}
  .metric-lbl  {font-size: 0.78rem; color: #94a3b8; margin-top: 4px;}
  .tip-box {
    background: rgba(99,102,241,.12); border-left: 3px solid #6366f1;
    padding: .6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: .4rem;
    color: #e2e8f0; font-size: .88rem;
  }
  div[data-testid="stDataFrameContainer"] {border: 1px solid #334155; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Session state helpers
# ────────────────────────────────────────────────────────────
def _init_state():
    for key, default in [
        ("raw_df",      None),
        ("clean_df",    None),
        ("report_raw",  None),
        ("report_clean",None),
        ("anomaly_mask",None),
        ("filename",    "dataset"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

_init_state()


# ────────────────────────────────────────────────────────────
# Sidebar — data upload / connection
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧹 DataCleaner Pro")
    st.markdown("---")
    st.markdown("### 📂 Load Data")
    tab_file, tab_sql = st.tabs(["File Upload", "SQL Database"])

    with tab_file:
        uploaded = st.file_uploader(
            "CSV / Excel / JSON",
            type=["csv", "xlsx", "xls", "json"],
        )
        if uploaded and st.button("Load File", use_container_width=True):
            with st.spinner("Reading file…"):
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                try:
                    df = load_auto(tmp_path)
                    st.session_state.raw_df      = df
                    st.session_state.clean_df    = df.copy()
                    st.session_state.report_raw  = build_quality_report(df)
                    st.session_state.report_clean= build_quality_report(df)
                    st.session_state.filename    = Path(uploaded.name).stem
                    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                finally:
                    os.unlink(tmp_path)

    with tab_sql:
        conn_str = st.text_input("Connection String",
            placeholder="sqlite:///mydb.db",
            help="SQLAlchemy URL")
        sql_q = st.text_area("Query / Table name",
            placeholder="SELECT * FROM sales",
            height=80)
        if st.button("Connect & Fetch", use_container_width=True):
            with st.spinner("Connecting…"):
                try:
                    df = load_auto("", connection_string=conn_str, sql_query=sql_q)
                    st.session_state.raw_df      = df
                    st.session_state.clean_df    = df.copy()
                    st.session_state.report_raw  = build_quality_report(df)
                    st.session_state.report_clean= build_quality_report(df)
                    st.session_state.filename    = "sql_export"
                    st.success(f"Fetched {len(df):,} rows")
                except Exception as e:
                    st.error(f"SQL error: {e}")

    if st.session_state.raw_df is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Cleaning Options")

        impute_strat = st.selectbox(
            "Impute missing values",
            ["auto", "mean", "median", "mode", "skip"],
        )
        fix_types_flag = st.checkbox("Auto-fix data types", value=True)
        dedup_flag     = st.checkbox("Remove duplicates", value=True)
        outlier_method = st.selectbox("Outlier method", ["iqr", "zscore", "none"])
        outlier_action = st.selectbox("Outlier action", ["cap", "remove", "skip"])
        run_iforest    = st.checkbox("Anomaly detection (Isolation Forest)", value=False)
        iforest_cont   = st.slider("Contamination", 0.01, 0.20, 0.05) if run_iforest else 0.05

        if st.button("🚀 Run Cleaning", use_container_width=True, type="primary"):
            with st.spinner("Cleaning in progress…"):
                cdf = st.session_state.raw_df.copy()
                if fix_types_flag:
                    cdf = fix_types(cdf)
                if impute_strat != "skip":
                    cdf = impute_missing(cdf, strategy=impute_strat)
                if dedup_flag:
                    cdf = remove_duplicates(cdf)
                if outlier_method != "none" and outlier_action != "skip":
                    cdf = handle_outliers(cdf, method=outlier_method, action=outlier_action)
                if run_iforest:
                    mask = detect_anomalies_isolation_forest(cdf, contamination=iforest_cont)
                    st.session_state.anomaly_mask = mask
                st.session_state.clean_df    = cdf
                st.session_state.report_clean= build_quality_report(cdf)
            st.success("✅ Cleaning complete!")


# ────────────────────────────────────────────────────────────
# Main area
# ────────────────────────────────────────────────────────────
if st.session_state.raw_df is None:
    # ── Welcome screen ─────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem'>
      <div style='font-size:5rem'>🧹</div>
      <h1 style='font-size:2.5rem;background:linear-gradient(90deg,#6366f1,#22d3ee);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 margin:.5rem 0'>DataCleaner Pro</h1>
      <p style='color:#94a3b8;font-size:1.1rem;max-width:500px;margin:auto'>
        Upload a CSV, Excel, or JSON file — or connect to a SQL database —
        then let the tool automatically clean, visualise, and export your data.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label in zip(
        [c1, c2, c3, c4],
        ["📥", "🔬", "📊", "📤"],
        ["Multi-format Input", "Auto Cleaning", "Interactive Viz", "Export Results"]
    ):
        col.markdown(
            f"<div class='metric-card'><div style='font-size:2rem'>{icon}</div>"
            f"<div class='metric-lbl' style='font-size:.9rem;color:#e2e8f0;margin-top:.4rem'>{label}</div></div>",
            unsafe_allow_html=True,
        )
    st.stop()


# ── Tabs ──────────────────────────────────────────────────
tab_overview, tab_data, tab_viz, tab_report, tab_export = st.tabs([
    "📋 Overview", "📄 Data Explorer", "📊 Visualise", "🧾 Quality Report", "📤 Export"
])

raw   = st.session_state.raw_df
clean = st.session_state.clean_df
rep_r = st.session_state.report_raw
rep_c = st.session_state.report_clean


# ── Overview ─────────────────────────────────────────────
with tab_overview:
    st.markdown("## 📋 Dataset Overview")
    r1, r2, r3, r4, r5 = st.columns(5)
    for col, val, lbl in zip(
        [r1, r2, r3, r4, r5],
        [
            rep_r["total_rows"],
            rep_r["total_cols"],
            rep_r["total_missing"],
            rep_r["duplicate_rows"],
            len(rep_r["numeric_cols"]),
        ],
        ["Total Rows", "Columns", "Missing Values", "Duplicates", "Numeric Cols"],
    ):
        col.markdown(
            f"<div class='metric-card'><div class='metric-val'>{val:,}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Missing Values")
        st.plotly_chart(plotly_missing_bar(rep_r), use_container_width=True, key="overview_missing")
    with col_r:
        st.markdown("### IQR Outliers")
        st.plotly_chart(plotly_outlier_bar(rep_r), use_container_width=True, key="overview_outliers")

    st.markdown("### 💡 Cleaning Suggestions")
    for tip in suggest_cleaning_strategies(rep_r):
        st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

    if st.session_state.anomaly_mask is not None:
        n_anom = int(st.session_state.anomaly_mask.sum())
        st.info(f"🤖 Isolation Forest detected **{n_anom}** anomalous rows "
                f"({n_anom/len(clean)*100:.1f}%)")


# ── Data Explorer ─────────────────────────────────────────
with tab_data:
    st.markdown("## 📄 Data Explorer")
    view_mode = st.radio("View", ["Raw data", "Cleaned data"], horizontal=True)
    df_view   = raw if view_mode == "Raw data" else clean

    col_filter = st.multiselect(
        "Filter columns (leave empty = all)",
        options=df_view.columns.tolist(),
    )
    display_df = df_view[col_filter] if col_filter else df_view
    st.dataframe(display_df, use_container_width=True, height=420)

    st.markdown(f"**Shape:** {display_df.shape[0]:,} rows × {display_df.shape[1]} cols")
    with st.expander("Descriptive Statistics"):
        st.dataframe(display_df.describe(include="all").T, use_container_width=True)

    if st.session_state.anomaly_mask is not None:
        with st.expander("🤖 Anomalous Rows (Isolation Forest)"):
            st.dataframe(clean[st.session_state.anomaly_mask], use_container_width=True)


# ── Visualise ─────────────────────────────────────────────
with tab_viz:
    st.markdown("## 📊 Visualisations")
    numeric_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
    categ_cols   = clean.select_dtypes(exclude=[np.number]).columns.tolist()
    all_cols     = clean.columns.tolist()

    viz_type = st.selectbox(
        "Chart type",
        ["Histogram", "Boxplot", "Bar Chart", "Pie Chart", "Correlation Heatmap"],
    )

    if viz_type in ("Histogram", "Boxplot"):
        if not numeric_cols:
            st.warning("No numeric columns available.")
        else:
            sel = st.selectbox("Select numeric column", numeric_cols)
            if viz_type == "Histogram":
                st.plotly_chart(plotly_histogram(clean, sel), use_container_width=True, key="viz_hist")
            else:
                st.plotly_chart(plotly_boxplot(clean, sel), use_container_width=True, key="viz_box")

    elif viz_type in ("Bar Chart", "Pie Chart"):
        if not all_cols:
            st.warning("No columns available.")
        else:
            sel   = st.selectbox("Select column", all_cols)
            top_n = st.slider("Top N values", 5, 30, 15)
            if viz_type == "Bar Chart":
                st.plotly_chart(plotly_bar(clean, sel, top_n=top_n), use_container_width=True, key="viz_bar")
            else:
                st.plotly_chart(plotly_pie(clean, sel, top_n=top_n), use_container_width=True, key="viz_pie")

    elif viz_type == "Correlation Heatmap":
        fig = plotly_correlation_heatmap(clean)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="viz_heatmap")
        else:
            st.warning("Need at least 2 numeric columns for a heatmap.")

    st.markdown("---")
    st.markdown("### 🔁 Compare Raw vs Cleaned")
    if numeric_cols:
        comp_col = st.selectbox("Column to compare", numeric_cols, key="compare_col")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Raw**")
            st.plotly_chart(plotly_histogram(raw, comp_col), use_container_width=True, key="compare_raw")
        with cb:
            st.markdown("**Cleaned**")
            st.plotly_chart(plotly_histogram(clean, comp_col), use_container_width=True, key="compare_clean")


# ── Quality Report ────────────────────────────────────────
with tab_report:
    st.markdown("## 🧾 Data Quality Report")
    mode = st.radio("Report for", ["Raw", "Cleaned"], horizontal=True)
    rep  = rep_r if mode == "Raw" else rep_c

    st.markdown("### Column Details")
    col_df = pd.DataFrame(rep["columns"])
    st.dataframe(col_df, use_container_width=True)

    st.markdown("### Cleaning Suggestions")
    for tip in suggest_cleaning_strategies(rep):
        st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

    # Generate & offer HTML report download
    if st.button("📄 Generate HTML Report"):
        with st.spinner("Building report…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                # static charts
                chart_paths = []
                for col in clean.select_dtypes(include=[np.number]).columns[:6]:
                    try:
                        chart_paths.append(plot_histogram_static(clean, col, tmpdir))
                        chart_paths.append(plot_boxplot_static(clean, col, tmpdir))
                    except Exception:
                        pass
                for col in clean.select_dtypes(exclude=[np.number]).columns[:3]:
                    try:
                        chart_paths.append(plot_bar_static(clean, col, tmpdir))
                    except Exception:
                        pass
                hmap = plot_correlation_heatmap_static(clean, tmpdir)
                if hmap:
                    chart_paths.append(hmap)

                html_path = os.path.join(tmpdir, "report.html")
                generate_html_report(
                    rep_c,
                    suggest_cleaning_strategies(rep_c),
                    chart_paths=chart_paths,
                    output_path=html_path,
                )
                with open(html_path, "r", encoding="utf-8") as f:
                    html_bytes = f.read().encode()

        st.download_button(
            "⬇️ Download HTML Report",
            data=html_bytes,
            file_name=f"{st.session_state.filename}_quality_report.html",
            mime="text/html",
        )


# ── Export ────────────────────────────────────────────────
with tab_export:
    st.markdown("## 📤 Export Cleaned Dataset")
    fmt = st.selectbox("Format", ["CSV", "Excel (.xlsx)", "JSON"])
    ext_map = {"CSV": ".csv", "Excel (.xlsx)": ".xlsx", "JSON": ".json"}
    ext     = ext_map[fmt]
    fname   = f"{st.session_state.filename}_cleaned{ext}"

    with st.spinner("Preparing export…"):
        buf = io.BytesIO()
        if ext == ".csv":
            clean.to_csv(buf, index=False)
            mime = "text/csv"
        elif ext == ".xlsx":
            clean.to_excel(buf, index=False, engine="openpyxl")
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            clean.to_json(buf, orient="records", indent=2)
            mime = "application/json"
        buf.seek(0)

    st.download_button(
        f"⬇️ Download {fmt}",
        data=buf,
        file_name=fname,
        mime=mime,
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")
    st.markdown("### 📊 Cleaning Summary")
    rows_removed = len(raw) - len(clean)
    null_removed = rep_r["total_missing"] - rep_c["total_missing"]
    dupe_removed = rep_r["duplicate_rows"] - rep_c["duplicate_rows"]

    m1, m2, m3 = st.columns(3)
    for col, val, lbl in zip(
        [m1, m2, m3],
        [rows_removed, null_removed, dupe_removed],
        ["Rows Removed", "Nulls Fixed", "Dupes Removed"],
    ):
        col.markdown(
            f"<div class='metric-card'><div class='metric-val'>{max(val,0):,}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )
