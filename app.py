"""
EconoPulse — Real-Time Economic Indicators & Policy Insight Platform
Main Streamlit application.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.forecasting import forecast_indicator
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

tab_overview, tab_compare, tab_signals, tab_forecast, tab_export = st.tabs(
    ["📊 Dashboard", "🌍 Country Comparison", "🚨 Signal Detection", "🔮 Forecasting", "💾 Export"]
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
