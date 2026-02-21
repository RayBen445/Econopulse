"""
EconoPulse — Real-Time Economic Indicators & Policy Insight Platform
Main Streamlit application.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.forecasting import forecast_indicator
from analytics.premium import (
    cluster_countries,
    compute_economic_health_score,
    compute_moving_average,
    compute_risk_scores,
    compute_volatility,
    compute_yoy_heatmap,
    detect_recessions,
    multi_country_forecast,
    rank_countries,
    run_custom_threshold_alerts,
)
from analytics.signals import alerts_to_dataframe, run_all_detectors
from data.fetcher import get_data, load_csv
from data.sample_data import INDICATORS, SAMPLE_COUNTRIES
from utils.export import generate_text_report, to_csv_bytes

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EconoPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper: severity badge colours
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {"high": "🔴", "medium": "🟡", "low": "🟢"}

# ---------------------------------------------------------------------------
# Sidebar — Data Source (must come before data load)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 EconoPulse")
    st.caption("Real-Time Economic Indicators & Policy Insight Platform")
    st.divider()

    data_source = st.radio(
        "Data Source",
        options=["Sample Data (Offline)", "World Bank API", "Upload CSV"],
        index=0,
    )

    csv_file = None
    if data_source == "Upload CSV":
        csv_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Columns required: year, country, indicator_code (or indicator_name), value",
        )

    st.divider()

    country_options = SAMPLE_COUNTRIES
    selected_countries = st.multiselect(
        "Countries",
        options=country_options,
        default=country_options[:3],
    )
    if not selected_countries:
        st.warning("Select at least one country.")
        selected_countries = country_options[:1]

    st.divider()


# ---------------------------------------------------------------------------
# Load data  (before the year-range slider so bounds are data-driven)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data…")
def load_data(source: str, csv_bytes: bytes | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "Upload CSV" and csv_bytes is not None:
        import io
        return get_data(source="csv", csv_file=io.BytesIO(csv_bytes))
    if source == "World Bank API":
        return get_data(source="worldbank")
    return get_data(source="sample")


csv_bytes = csv_file.read() if csv_file else None
if csv_file:
    csv_file.seek(0)

indicators_df, fx_df = load_data(data_source, csv_bytes)

# Derive year bounds from the actual data
_data_years = sorted(indicators_df["year"].dropna().unique()) if not indicators_df.empty else [2000, 2023]
_year_min, _year_max = int(_data_years[0]), int(_data_years[-1])

# Sidebar continued — year range and indicator (depend on loaded data)
with st.sidebar:
    year_range = st.slider(
        "Year Range",
        min_value=_year_min,
        max_value=_year_max,
        value=(_year_min + min(5, _year_max - _year_min), _year_max),
    )

    indicator_options = {v: k for k, v in INDICATORS.items()}
    selected_indicator_name = st.selectbox(
        "Primary Indicator",
        options=list(indicator_options.keys()),
        index=0,
    )
    selected_indicator_code = indicator_options[selected_indicator_name]

    st.divider()
    st.caption("EconoPulse v1.0 · Data: World Bank / Sample")


# Apply filters
filtered_df = indicators_df[
    (indicators_df["country"].isin(selected_countries))
    & (indicators_df["year"].between(*year_range))
].copy()

fx_filtered = fx_df[
    (fx_df["country"].isin(selected_countries))
    & (fx_df["year"].between(*year_range))
].copy()

# Run signal detection (on full dataset for completeness, then filter for display)
all_alerts = run_all_detectors(
    indicators_df[indicators_df["country"].isin(selected_countries)]
)
alerts_df = alerts_to_dataframe(all_alerts)
alerts_filtered = alerts_df[alerts_df["year"].between(*year_range)] if not alerts_df.empty else alerts_df

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📈 EconoPulse")
st.subheader("Real-Time Economic Indicators & Policy Insight Platform")

high_count = len(alerts_filtered[alerts_filtered["severity"] == "high"]) if not alerts_filtered.empty else 0
medium_count = len(alerts_filtered[alerts_filtered["severity"] == "medium"]) if not alerts_filtered.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries Analysed", len(selected_countries))
col2.metric("Years Covered", f"{year_range[0]}–{year_range[1]}")
col3.metric("🔴 High Alerts", high_count)
col4.metric("🟡 Medium Alerts", medium_count)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_compare, tab_signals, tab_forecast, tab_export, tab_premium = st.tabs(
    ["📊 Dashboard", "🌍 Country Comparison", "🚨 Signal Detection", "🔮 Forecasting", "💾 Export", "⭐ Premium"]
)


# ===========================================================================
# Tab 1 — Dashboard / Overview
# ===========================================================================

with tab_overview:
    st.header("Economic Indicators Dashboard")

    # Summary KPI cards per country for the latest year in range
    latest_year = filtered_df["year"].max() if not filtered_df.empty else year_range[1]

    def latest_value(country: str, code: str) -> str:
        subset = filtered_df[
            (filtered_df["country"] == country) & (filtered_df["indicator_code"] == code)
        ]
        if subset.empty:
            return "N/A"
        val = subset.loc[subset["year"].idxmax(), "value"]
        return f"{val:.2f}"

    kpi_codes = list(INDICATORS.keys())[:3]  # top 3 indicators for KPI display
    for country in selected_countries:
        with st.expander(f"📌 {country} — Latest Indicators ({int(latest_year)})", expanded=True):
            k_cols = st.columns(len(kpi_codes))
            for col, code in zip(k_cols, kpi_codes):
                col.metric(INDICATORS[code].split(" (")[0], latest_value(country, code))

    st.divider()

    # Interactive time-series chart for selected indicator
    st.subheader(f"Trend: {selected_indicator_name}")
    chart_df = filtered_df[filtered_df["indicator_code"] == selected_indicator_code]

    if chart_df.empty:
        st.info("No data for this indicator in the selected filters.")
    else:
        fig = px.line(
            chart_df,
            x="year",
            y="value",
            color="country",
            markers=True,
            labels={"value": selected_indicator_name, "year": "Year"},
            title=f"{selected_indicator_name} — {year_range[0]}–{year_range[1]}",
        )
        fig.update_layout(legend_title="Country", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # FX rates chart
    if not fx_filtered.empty:
        st.subheader("FX Rate (Local Currency per USD)")
        fig_fx = px.line(
            fx_filtered,
            x="year",
            y="fx_rate",
            color="country",
            markers=True,
            labels={"fx_rate": "FX Rate (LCU/USD)", "year": "Year"},
            title=f"FX Rates — {year_range[0]}–{year_range[1]}",
        )
        fig_fx.update_layout(legend_title="Country", hovermode="x unified")
        st.plotly_chart(fig_fx, use_container_width=True)

    # All indicators grid
    st.subheader("All Indicators Overview")
    n_cols = 2
    codes = list(INDICATORS.keys())
    for i in range(0, len(codes), n_cols):
        row_cols = st.columns(n_cols)
        for col, code in zip(row_cols, codes[i : i + n_cols]):
            name = INDICATORS[code]
            subset = filtered_df[filtered_df["indicator_code"] == code]
            if subset.empty:
                col.caption(f"{name}: no data")
                continue
            fig_small = px.line(
                subset,
                x="year",
                y="value",
                color="country",
                markers=False,
                labels={"value": name, "year": "Year"},
                title=name,
                height=300,
            )
            fig_small.update_layout(showlegend=True, legend_title="", margin={"t": 40})
            col.plotly_chart(fig_small, use_container_width=True)


# ===========================================================================
# Tab 2 — Country Comparison
# ===========================================================================

with tab_compare:
    st.header("Country Comparison")

    if len(selected_countries) < 2:
        st.info("Select at least two countries in the sidebar to compare.")
    else:
        cmp_indicator = st.selectbox(
            "Indicator to compare",
            options=list(INDICATORS.values()),
            index=0,
            key="cmp_indicator",
        )
        cmp_code = {v: k for k, v in INDICATORS.items()}[cmp_indicator]
        cmp_df = filtered_df[filtered_df["indicator_code"] == cmp_code]

        if cmp_df.empty:
            st.info("No data for this indicator.")
        else:
            # Line chart
            fig_cmp = px.line(
                cmp_df,
                x="year",
                y="value",
                color="country",
                markers=True,
                labels={"value": cmp_indicator, "year": "Year"},
                title=f"{cmp_indicator}: {' vs '.join(selected_countries)}",
            )
            fig_cmp.update_layout(hovermode="x unified")
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Bar chart: latest year comparison
            latest_cmp = cmp_df[cmp_df["year"] == cmp_df["year"].max()]
            fig_bar = px.bar(
                latest_cmp,
                x="country",
                y="value",
                color="country",
                labels={"value": cmp_indicator, "country": "Country"},
                title=f"{cmp_indicator} — Latest Year ({int(cmp_df['year'].max())})",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Correlation heatmap (if ≥2 indicators selected)
            st.subheader("Cross-Indicator Correlation (per country)")
            for country in selected_countries:
                pivot = (
                    filtered_df[filtered_df["country"] == country]
                    .pivot_table(index="year", columns="indicator_name", values="value")
                    .dropna()
                )
                if pivot.shape[1] >= 2:
                    corr = pivot.corr()
                    fig_heat = px.imshow(
                        corr,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        title=f"Correlation Matrix — {country}",
                        height=350,
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)


# ===========================================================================
# Tab 3 — Signal Detection
# ===========================================================================

with tab_signals:
    st.header("🚨 Economic Stress Signal Detection")
    st.caption(
        "Signals are flagged based on absolute level thresholds and year-over-year changes."
    )

    if alerts_filtered.empty:
        st.success("✅ No stress signals detected for the selected filters.")
    else:
        # Summary counts
        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("Total Signals", len(alerts_filtered))
        s_col2.metric("🔴 High", len(alerts_filtered[alerts_filtered["severity"] == "high"]))
        s_col3.metric("🟡 Medium", len(alerts_filtered[alerts_filtered["severity"] == "medium"]))

        # Severity filter
        severity_filter = st.multiselect(
            "Filter by severity",
            options=["high", "medium", "low"],
            default=["high", "medium"],
            key="sev_filter",
        )
        display_alerts = alerts_filtered[alerts_filtered["severity"].isin(severity_filter)]

        # Country filter
        country_filter = st.multiselect(
            "Filter by country",
            options=sorted(alerts_filtered["country"].unique()),
            default=sorted(alerts_filtered["country"].unique()),
            key="country_filter",
        )
        display_alerts = display_alerts[display_alerts["country"].isin(country_filter)]

        if display_alerts.empty:
            st.info("No signals match the selected filters.")
        else:
            # Signal timeline
            fig_timeline = px.scatter(
                display_alerts,
                x="year",
                y="country",
                color="severity",
                symbol="severity",
                hover_data=["signal_type", "value", "description"],
                color_discrete_map={"high": "#E74C3C", "medium": "#F39C12", "low": "#2ECC71"},
                labels={"year": "Year", "country": "Country"},
                title="Signal Timeline",
                height=400,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Alerts table
            st.subheader("Alert Details")
            styled = display_alerts[
                ["country", "year", "indicator", "signal_type", "severity", "value", "description"]
            ].sort_values(["severity", "country", "year"])
            st.dataframe(styled, use_container_width=True)

    # Narrative policy summary
    st.divider()
    st.subheader("📝 Policy Narrative Summary")
    for country in selected_countries:
        country_alerts_c = alerts_df[alerts_df["country"] == country] if not alerts_df.empty else pd.DataFrame()
        country_data = indicators_df[
            (indicators_df["country"] == country)
            & (indicators_df["year"].between(*year_range))
        ]

        high_alerts = country_alerts_c[country_alerts_c["severity"] == "high"] if not country_alerts_c.empty else pd.DataFrame()
        med_alerts = country_alerts_c[country_alerts_c["severity"] == "medium"] if not country_alerts_c.empty else pd.DataFrame()

        with st.expander(f"📋 {country} — Policy Narrative"):
            if country_data.empty:
                st.write("No data available for narrative generation.")
                continue

            # Latest indicator values
            latest_year_n = country_data["year"].max()
            narratives = []
            for code, name in INDICATORS.items():
                latest = country_data[
                    (country_data["indicator_code"] == code)
                    & (country_data["year"] == latest_year_n)
                ]
                if not latest.empty:
                    narratives.append(f"- **{name}**: {latest.iloc[0]['value']:.2f}")

            if narratives:
                st.markdown(f"**Latest indicators ({int(latest_year_n)}):**")
                for n in narratives:
                    st.markdown(n)

            if not high_alerts.empty:
                st.error(
                    f"⚠️ {len(high_alerts)} high-severity signal(s) detected: "
                    + "; ".join(high_alerts["signal_type"].unique())
                )
            if not med_alerts.empty:
                st.warning(
                    f"⚡ {len(med_alerts)} medium-severity signal(s): "
                    + "; ".join(med_alerts["signal_type"].unique())
                )
            if high_alerts.empty and med_alerts.empty:
                st.success("No major stress signals detected for this country.")


# ===========================================================================
# Tab 4 — Forecasting
# ===========================================================================

with tab_forecast:
    st.header("🔮 Economic Indicator Forecasting")
    st.caption("Uses Holt's linear trend model (or linear regression as fallback).")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        fc_country = st.selectbox("Country", options=selected_countries, key="fc_country")
    with f_col2:
        fc_indicator_name = st.selectbox(
            "Indicator",
            options=list(INDICATORS.values()),
            index=0,
            key="fc_indicator",
        )
        fc_indicator_code = {v: k for k, v in INDICATORS.items()}[fc_indicator_name]
    with f_col3:
        n_periods = st.slider("Forecast Horizon (years)", min_value=1, max_value=10, value=5)

    fc_df = forecast_indicator(
        indicators_df,
        country=fc_country,
        indicator_code=fc_indicator_code,
        n_periods=n_periods,
    )

    if fc_df is None:
        st.warning("Not enough data to generate a forecast (need at least 4 observations).")
    else:
        historical_fc = fc_df[fc_df["type"] == "historical"]
        forecasted_fc = fc_df[fc_df["type"] == "forecast"]

        fig_fc = go.Figure()
        fig_fc.add_trace(
            go.Scatter(
                x=historical_fc["year"],
                y=historical_fc["value"],
                mode="lines+markers",
                name="Historical",
                line={"color": "#2196F3"},
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=forecasted_fc["year"],
                y=forecasted_fc["value"],
                mode="lines+markers",
                name="Forecast",
                line={"color": "#FF9800", "dash": "dash"},
            )
        )
        fig_fc.update_layout(
            title=f"{fc_indicator_name} Forecast — {fc_country}",
            xaxis_title="Year",
            yaxis_title=fc_indicator_name,
            hovermode="x unified",
            legend_title="Type",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table
        st.subheader("Forecast Values")
        st.dataframe(
            forecasted_fc[["year", "value"]].rename(
                columns={"year": "Year", "value": "Forecasted Value"}
            ).round(2),
            use_container_width=True,
        )


# ===========================================================================
# Tab 5 — Export
# ===========================================================================

with tab_export:
    st.header("💾 Export Data & Reports")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.subheader("📥 Download Filtered Data (CSV)")
        csv_data = to_csv_bytes(filtered_df)
        st.download_button(
            label="Download Indicators CSV",
            data=csv_data,
            file_name=f"econopulse_indicators_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv",
        )

        if not fx_filtered.empty:
            fx_csv = to_csv_bytes(fx_filtered)
            st.download_button(
                label="Download FX Rates CSV",
                data=fx_csv,
                file_name=f"econopulse_fx_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
            )

        if not alerts_filtered.empty:
            alerts_csv = to_csv_bytes(alerts_filtered)
            st.download_button(
                label="Download Alerts CSV",
                data=alerts_csv,
                file_name=f"econopulse_alerts_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
            )

    with exp_col2:
        st.subheader("📄 Download Text Report")
        report_text = generate_text_report(
            indicators_df=indicators_df,
            alerts_df=alerts_df,
            selected_countries=selected_countries,
            selected_year_range=year_range,
        )
        st.text_area("Report Preview", value=report_text, height=300)
        st.download_button(
            label="Download Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"econopulse_report_{year_range[0]}_{year_range[1]}.txt",
            mime="text/plain",
        )

    st.divider()
    st.subheader("📊 Data Preview")
    st.dataframe(filtered_df.head(50), use_container_width=True)


# ===========================================================================
# Tab 6 — Premium Features
# ===========================================================================

with tab_premium:
    st.header("⭐ Premium Analytics")
    st.caption(
        "Ten advanced analytical features built on top of the core EconoPulse data pipeline."
    )

    (
        ptab1, ptab2, ptab3, ptab4, ptab5,
        ptab6, ptab7, ptab8, ptab9, ptab10,
    ) = st.tabs([
        "1 Health Score",
        "2 Multi-Forecast",
        "3 YoY Heatmap",
        "4 Risk Heatmap",
        "5 Recessions",
        "6 Rankings",
        "7 Moving Avg",
        "8 Custom Alerts",
        "9 Volatility",
        "10 Clustering",
    ])

    # ── 1. Economic Health Score ─────────────────────────────────────────────
    with ptab1:
        st.subheader("📊 Economic Health Score")
        st.caption(
            "Composite 0–100 index: GDP growth and GDP per capita pull the score up; "
            "inflation and unemployment pull it down. Weights: GDP growth 30 %, "
            "Inflation 25 %, Unemployment 25 %, GDP per capita 20 %."
        )
        health_df = compute_economic_health_score(filtered_df)
        if health_df.empty:
            st.info("Not enough indicator data to compute health scores.")
        else:
            fig_health = px.line(
                health_df,
                x="year",
                y="health_score",
                color="country",
                markers=True,
                labels={"health_score": "Health Score (0–100)", "year": "Year"},
                title=f"Economic Health Score — {year_range[0]}–{year_range[1]}",
            )
            fig_health.update_layout(hovermode="x unified", yaxis_range=[0, 100])
            st.plotly_chart(fig_health, use_container_width=True)

            latest_health = (
                health_df.sort_values("year")
                .groupby("country")
                .last()
                .reset_index()[["country", "year", "health_score"]]
                .sort_values("health_score", ascending=False)
            )
            st.subheader(f"Latest Health Scores ({int(latest_health['year'].max())})")
            st.dataframe(latest_health.rename(columns={
                "country": "Country", "year": "Year", "health_score": "Health Score"
            }), use_container_width=True)

    # ── 2. Multi-Country Forecast Comparison ─────────────────────────────────
    with ptab2:
        st.subheader("🔮 Multi-Country Forecast Comparison")
        st.caption(
            "Forecast the same indicator for all selected countries on one chart "
            "using Holt's linear trend (or OLS fallback)."
        )
        p2_col1, p2_col2 = st.columns(2)
        with p2_col1:
            p2_indicator_name = st.selectbox(
                "Indicator",
                options=list(INDICATORS.values()),
                index=0,
                key="p2_indicator",
            )
            p2_indicator_code = {v: k for k, v in INDICATORS.items()}[p2_indicator_name]
        with p2_col2:
            p2_horizon = st.slider("Forecast Horizon (years)", 1, 10, 5, key="p2_horizon")

        mc_fc = multi_country_forecast(
            indicators_df,
            countries=selected_countries,
            indicator_code=p2_indicator_code,
            n_periods=p2_horizon,
        )
        if mc_fc.empty:
            st.warning("Insufficient data to forecast any of the selected countries.")
        else:
            historical_mc = mc_fc[mc_fc["type"] == "historical"]
            forecast_mc = mc_fc[mc_fc["type"] == "forecast"]
            fig_mc = go.Figure()
            colours = px.colors.qualitative.Plotly
            for idx, country in enumerate(mc_fc["country"].unique()):
                colour = colours[idx % len(colours)]
                hist_c = historical_mc[historical_mc["country"] == country]
                fc_c = forecast_mc[forecast_mc["country"] == country]
                fig_mc.add_trace(go.Scatter(
                    x=hist_c["year"], y=hist_c["value"],
                    mode="lines+markers", name=f"{country} (historical)",
                    line={"color": colour},
                ))
                fig_mc.add_trace(go.Scatter(
                    x=fc_c["year"], y=fc_c["value"],
                    mode="lines+markers", name=f"{country} (forecast)",
                    line={"color": colour, "dash": "dash"},
                ))
            fig_mc.update_layout(
                title=f"{p2_indicator_name} — Multi-Country Forecast",
                xaxis_title="Year",
                yaxis_title=p2_indicator_name,
                hovermode="x unified",
            )
            st.plotly_chart(fig_mc, use_container_width=True)

    # ── 3. YoY Change Heatmap ─────────────────────────────────────────────────
    with ptab3:
        st.subheader("🌡️ Year-over-Year Change Heatmap")
        st.caption(
            "Each cell shows the absolute year-over-year change (percentage points) "
            "for the selected indicator."
        )
        p3_indicator_name = st.selectbox(
            "Indicator",
            options=list(INDICATORS.values()),
            index=0,
            key="p3_indicator",
        )
        p3_code = {v: k for k, v in INDICATORS.items()}[p3_indicator_name]
        yoy_pivot = compute_yoy_heatmap(filtered_df, p3_code, countries=selected_countries)
        if yoy_pivot.empty:
            st.info("Not enough data for a YoY heatmap.")
        else:
            fig_yoy = px.imshow(
                yoy_pivot,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                labels={"x": "Year", "y": "Country", "color": "YoY Change (pp)"},
                title=f"YoY Change — {p3_indicator_name}",
                aspect="auto",
            )
            fig_yoy.update_xaxes(tickangle=45)
            st.plotly_chart(fig_yoy, use_container_width=True)

    # ── 4. Economic Risk Heatmap ──────────────────────────────────────────────
    with ptab4:
        st.subheader("🔥 Economic Risk Score Heatmap")
        st.caption(
            "Annual risk score per country = sum of alert severity weights "
            "(low = 1, medium = 2, high = 3). Higher score = more economic stress."
        )
        risk_pivot = compute_risk_scores(alerts_df, countries=selected_countries)
        if risk_pivot.empty:
            st.success("✅ No stress signals detected — risk scores are zero.")
        else:
            fig_risk = px.imshow(
                risk_pivot,
                color_continuous_scale="YlOrRd",
                labels={"x": "Year", "y": "Country", "color": "Risk Score"},
                title="Economic Risk Score Heatmap",
                aspect="auto",
            )
            fig_risk.update_xaxes(tickangle=45)
            st.plotly_chart(fig_risk, use_container_width=True)

    # ── 5. Recession Detection ────────────────────────────────────────────────
    with ptab5:
        st.subheader("📉 Recession Detection")
        st.caption(
            "A recession is defined as two or more consecutive years of negative "
            "GDP growth. The chart highlights these periods as shaded bands."
        )
        p5_country = st.selectbox("Country", options=selected_countries, key="p5_country")
        p5_min_years = st.slider("Minimum consecutive years", 1, 5, 2, key="p5_min")

        gdp_series = filtered_df[
            (filtered_df["country"] == p5_country)
            & (filtered_df["indicator_code"] == "NY.GDP.MKTP.KD.ZG")
        ].sort_values("year")

        recessions = detect_recessions(
            indicators_df, p5_country, min_consecutive=p5_min_years
        )

        if gdp_series.empty:
            st.info("No GDP data available for this country.")
        else:
            fig_rec = go.Figure()
            fig_rec.add_trace(go.Scatter(
                x=gdp_series["year"],
                y=gdp_series["value"],
                mode="lines+markers",
                name="GDP Growth (%)",
                line={"color": "#2196F3"},
            ))
            fig_rec.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="0 %")
            for rec in recessions:
                fig_rec.add_vrect(
                    x0=rec["start_year"] - 0.5,
                    x1=rec["end_year"] + 0.5,
                    fillcolor="red",
                    opacity=0.15,
                    line_width=0,
                    annotation_text="Recession",
                    annotation_position="top left",
                )
            fig_rec.update_layout(
                title=f"GDP Growth & Recession Periods — {p5_country}",
                xaxis_title="Year",
                yaxis_title="GDP Growth (% annual)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_rec, use_container_width=True)

            if recessions:
                st.subheader("Detected Recession Periods")
                rec_df = pd.DataFrame(recessions)
                rec_df["duration"] = rec_df["end_year"] - rec_df["start_year"] + 1
                st.dataframe(
                    rec_df.rename(columns={
                        "country": "Country", "start_year": "Start",
                        "end_year": "End", "depth": "Avg GDP Growth (%)",
                        "duration": "Duration (yrs)",
                    }),
                    use_container_width=True,
                )
            else:
                st.success(f"No recession periods detected for {p5_country}.")

    # ── 6. Country Percentile Rankings ───────────────────────────────────────
    with ptab6:
        st.subheader("🏆 Country Percentile Rankings")
        st.caption(
            "Rank selected countries against each other for a given indicator and year. "
            "Percentile 100 = best performer."
        )
        p6_col1, p6_col2 = st.columns(2)
        with p6_col1:
            p6_indicator_name = st.selectbox(
                "Indicator",
                options=list(INDICATORS.values()),
                index=0,
                key="p6_indicator",
            )
            p6_code = {v: k for k, v in INDICATORS.items()}[p6_indicator_name]
        with p6_col2:
            p6_year = st.selectbox(
                "Year",
                options=sorted(filtered_df["year"].unique(), reverse=True),
                key="p6_year",
            )

        rankings = rank_countries(filtered_df, p6_code, year=int(p6_year))
        if rankings.empty:
            st.info("No data available for ranking.")
        else:
            fig_rank = px.bar(
                rankings,
                x="country",
                y="percentile",
                color="percentile",
                color_continuous_scale="RdYlGn",
                labels={"percentile": "Percentile", "country": "Country"},
                title=f"{p6_indicator_name} Percentile Rankings ({int(p6_year)})",
                text="rank",
            )
            fig_rank.update_traces(textposition="outside")
            fig_rank.update_layout(yaxis_range=[0, 110], coloraxis_showscale=False)
            st.plotly_chart(fig_rank, use_container_width=True)
            st.dataframe(
                rankings.rename(columns={
                    "country": "Country", "year": "Year",
                    "value": "Value", "rank": "Rank", "percentile": "Percentile",
                }),
                use_container_width=True,
            )

    # ── 7. Moving Average Trend Overlay ──────────────────────────────────────
    with ptab7:
        st.subheader("📈 Moving Average Trend Overlay")
        st.caption(
            "Overlay a rolling mean on top of the raw indicator series to smooth "
            "out short-term noise and highlight medium-term trends."
        )
        p7_col1, p7_col2, p7_col3 = st.columns(3)
        with p7_col1:
            p7_country = st.selectbox("Country", options=selected_countries, key="p7_country")
        with p7_col2:
            p7_indicator_name = st.selectbox(
                "Indicator",
                options=list(INDICATORS.values()),
                index=0,
                key="p7_indicator",
            )
            p7_code = {v: k for k, v in INDICATORS.items()}[p7_indicator_name]
        with p7_col3:
            p7_window = st.slider("Window (years)", 2, 10, 3, key="p7_window")

        ma_df = compute_moving_average(filtered_df, p7_country, p7_code, window=p7_window)
        if ma_df.empty:
            st.info("No data available for moving average.")
        else:
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=ma_df["year"], y=ma_df["value"],
                mode="lines+markers", name="Raw",
                line={"color": "#90CAF9"},
            ))
            fig_ma.add_trace(go.Scatter(
                x=ma_df["year"], y=ma_df["moving_avg"],
                mode="lines", name=f"{p7_window}-year MA",
                line={"color": "#1565C0", "width": 3},
            ))
            fig_ma.update_layout(
                title=f"{p7_indicator_name} — {p7_window}-Year Moving Average ({p7_country})",
                xaxis_title="Year",
                yaxis_title=p7_indicator_name,
                hovermode="x unified",
            )
            st.plotly_chart(fig_ma, use_container_width=True)

    # ── 8. Custom Alert Thresholds ────────────────────────────────────────────
    with ptab8:
        st.subheader("🎛️ Custom Alert Thresholds")
        st.caption(
            "Set your own signal thresholds for each indicator. "
            "Alerts fire when a value exceeds (or falls below, for GDP growth) "
            "the specified threshold."
        )
        custom_thresholds: dict[str, float] = {}
        th_col1, th_col2 = st.columns(2)
        with th_col1:
            infl_th = st.slider(
                "Inflation threshold (%)", 0.0, 50.0, 8.0, 0.5, key="th_infl"
            )
            custom_thresholds["FP.CPI.TOTL.ZG"] = infl_th

            gdp_th = st.slider(
                "GDP growth floor (%)", -10.0, 5.0, 0.0, 0.5, key="th_gdp"
            )
            custom_thresholds["NY.GDP.MKTP.KD.ZG"] = gdp_th
        with th_col2:
            unemp_th = st.slider(
                "Unemployment threshold (%)", 0.0, 40.0, 12.0, 0.5, key="th_unemp"
            )
            custom_thresholds["SL.UEM.TOTL.ZS"] = unemp_th

        custom_alerts = run_custom_threshold_alerts(filtered_df, custom_thresholds)
        if custom_alerts.empty:
            st.success("✅ No custom alerts triggered with the current thresholds.")
        else:
            st.metric("Custom Alerts Triggered", len(custom_alerts))
            st.dataframe(
                custom_alerts[
                    ["country", "year", "indicator", "value", "signal_type", "description"]
                ].sort_values(["country", "year"]),
                use_container_width=True,
            )

    # ── 9. Volatility Analysis ────────────────────────────────────────────────
    with ptab9:
        st.subheader("📊 Indicator Volatility Analysis")
        st.caption(
            "Rolling standard deviation of the selected indicator reveals which "
            "countries have experienced the most erratic behavior over time."
        )
        p9_col1, p9_col2 = st.columns(2)
        with p9_col1:
            p9_indicator_name = st.selectbox(
                "Indicator",
                options=list(INDICATORS.values()),
                index=0,
                key="p9_indicator",
            )
            p9_code = {v: k for k, v in INDICATORS.items()}[p9_indicator_name]
        with p9_col2:
            p9_window = st.slider("Rolling window (years)", 2, 10, 5, key="p9_window")

        vol_df = compute_volatility(
            filtered_df, p9_code, window=p9_window, countries=selected_countries
        )
        if vol_df.empty or vol_df["rolling_std"].isna().all():
            st.info("Not enough data to compute volatility.")
        else:
            fig_vol = px.line(
                vol_df.dropna(subset=["rolling_std"]),
                x="year",
                y="rolling_std",
                color="country",
                markers=True,
                labels={
                    "rolling_std": f"{p9_window}-yr Rolling Std Dev",
                    "year": "Year",
                },
                title=(
                    f"{p9_indicator_name} — {p9_window}-Year Rolling Volatility"
                ),
            )
            fig_vol.update_layout(hovermode="x unified")
            st.plotly_chart(fig_vol, use_container_width=True)

            avg_vol = (
                vol_df.groupby("country")["rolling_std"]
                .mean()
                .reset_index()
                .rename(columns={"rolling_std": "Avg Volatility"})
                .sort_values("Avg Volatility", ascending=False)
            )
            st.subheader("Average Volatility by Country")
            fig_vol_bar = px.bar(
                avg_vol,
                x="country",
                y="Avg Volatility",
                color="Avg Volatility",
                color_continuous_scale="Reds",
                labels={"country": "Country"},
                title=f"Average {p9_window}-Year Volatility — {p9_indicator_name}",
            )
            fig_vol_bar.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_vol_bar, use_container_width=True)

    # ── 10. Economic Similarity Clustering ────────────────────────────────────
    with ptab10:
        st.subheader("🔵 Economic Similarity Clustering")
        st.caption(
            "K-means clustering groups countries by similarity across all indicators "
            "in a selected year (features are standardised before clustering)."
        )
        p10_col1, p10_col2 = st.columns(2)
        with p10_col1:
            p10_year = st.selectbox(
                "Year",
                options=sorted(filtered_df["year"].unique(), reverse=True),
                key="p10_year",
            )
        with p10_col2:
            p10_k = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=min(6, len(selected_countries)),
                value=min(3, len(selected_countries)),
                key="p10_k",
            )

        cluster_df = cluster_countries(filtered_df, year=int(p10_year), n_clusters=p10_k)
        if cluster_df.empty:
            st.info("Not enough data to cluster countries for the selected year.")
        else:
            indicator_cols = [c for c in cluster_df.columns
                              if c not in ("country", "cluster", "year")]
            if len(indicator_cols) >= 2:
                x_col, y_col = indicator_cols[0], indicator_cols[1]
                fig_cluster = px.scatter(
                    cluster_df,
                    x=x_col,
                    y=y_col,
                    color=cluster_df["cluster"].astype(str),
                    text="country",
                    labels={
                        x_col: INDICATORS.get(x_col, x_col),
                        y_col: INDICATORS.get(y_col, y_col),
                        "color": "Cluster",
                    },
                    title=f"Country Clusters ({int(p10_year)}) — k={p10_k}",
                    height=450,
                )
                fig_cluster.update_traces(textposition="top center", marker_size=12)
                st.plotly_chart(fig_cluster, use_container_width=True)

            cluster_display = cluster_df[["country", "cluster"]].copy()
            cluster_display["cluster"] = cluster_display["cluster"].apply(
                lambda c: f"Cluster {c + 1}"
            )
            st.subheader("Cluster Assignments")
            st.dataframe(
                cluster_display.rename(columns={"country": "Country", "cluster": "Cluster"})
                .sort_values(["Cluster", "Country"]),
                use_container_width=True,
            )
