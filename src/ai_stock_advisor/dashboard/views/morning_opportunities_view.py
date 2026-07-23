"""
Morning Opportunities View Panel.
Displays the Top 10 pre-market F&O opportunities with filters, details cards, and manual scan triggers.
"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from config.settings import settings
from ai_stock_advisor.scanner.morning_scanner import MorningOpportunityScanner

# We can cache the scanner instance to avoid duplicate db init logs
@st.cache_resource
def get_scanner() -> MorningOpportunityScanner:
    return MorningOpportunityScanner()


def render_morning_opportunities() -> None:
    """Renders the Morning Opportunities page containing the Top 10 opportunities dashboard."""
    st.markdown("<h2 class='sub-title'>🌅 Morning AI F&O Opportunity Scanner</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Ranked pre-market derivative trading opportunities compiled daily at 08:45 AM IST using multi-timeframe indicators, option chains, news sentiment, and risk levels.</p>",
        unsafe_allow_html=True,
    )

    json_path = Path(settings.BASE_DIR) / "data" / "morning_opportunities.json"
    scanner = get_scanner()

    # Manual Scanner Run Button
    col_btn, col_spacer = st.columns([1.5, 8.5])
    if col_btn.button("🔄 Run Pre-Market Scan", use_container_width=True):
        with st.spinner("Executing NSE F&O multi-timeframe scan, options PCR, news analyzer, and risk engines..."):
            try:
                scanner.run_scan()
                st.success("✅ Morning pre-market scan complete! Data updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Scan failed: {exc}")

    # Load Data
    if not json_path.exists():
        st.info("⚠️ Morning opportunities data is not generated yet. Click the 'Run Pre-Market Scan' button above to generate the pre-market setups.")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        st.error(f"Failed loading morning opportunities cache file: {exc}")
        return

    if not data:
        st.warning("No opportunities found in the daily pre-market records.")
        return

    df = pd.DataFrame(data)

    # 1. Filters & Search Grid
    st.markdown("### 🔍 Filters & Search")
    col_search, col_trend, col_risk = st.columns(3)
    
    search_ticker = col_search.text_input("Search Ticker", "").strip().upper()
    
    trends = df["Trend"].unique().tolist()
    selected_trends = col_trend.multiselect("Trend Direction", trends, default=trends)
    
    risk_levels = df["Risk_Level"].unique().tolist()
    selected_risks = col_risk.multiselect("Risk Profile", risk_levels, default=risk_levels)

    # Filter dataframe
    filtered_df = df[
        df["Ticker"].str.contains(search_ticker, case=False, na=False) &
        df["Trend"].isin(selected_trends) &
        df["Risk_Level"].isin(selected_risks)
    ]

    # Convert to presentation format
    display_cols = [
        "Ticker", "Price", "Trend", "Entry", "Stop_Loss", "Target_1", 
        "Target_2", "Target_3", "Risk_Reward", "Probability", "Confidence", "Risk_Level"
    ]
    
    table_df = filtered_df[display_cols].copy()
    table_df.columns = [
        "Ticker", "Price (₹)", "Trend", "Entry (₹)", "Stop Loss (₹)", "Target 1 (₹)",
        "Target 2 (₹)", "Target 3 (₹)", "R:R Ratio", "Score (%)", "Confidence (%)", "Risk Level"
    ]

    # Render Table
    st.markdown("### 🏆 Top Opportunities Grid")
    st.dataframe(
        table_df.style.map(
            lambda x: "color: #10B981; font-weight: bold;" if x == "BULLISH" else ("color: #EF4444; font-weight: bold;" if x == "BEARISH" else ""),
            subset=["Trend"]
        ),
        use_container_width=True,
        hide_index=True
    )

    # CSV Export Button
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Opportunities as CSV",
        data=csv_data,
        file_name="morning_opportunities.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # 2. Detailed Breakdown Panel
    st.markdown("### 📊 Interactive Setup Analysis")
    selected_ticker = st.selectbox("Select Ticker for Detailed Trade Card", filtered_df["Ticker"].tolist())
    
    if selected_ticker:
        row = filtered_df[filtered_df["Ticker"] == selected_ticker].iloc[0]
        
        # Load individual metrics from database scan records if available to render gauges and details
        # For simplicity, we render details directly from the opportunity record keys!
        
        col_gauge, col_trade = st.columns([4, 6])
        
        with col_gauge:
            # Render visual probability details
            st.markdown(
                f"""
                <div class='metric-card' style='text-align: center;'>
                    <h4 style='color:#94A3B8; margin:0;'>Breakout Probability</h4>
                    <h1 style='font-size: 3.5rem; color:#00E5FF; margin:10px 0;'>{row['Probability']}%</h1>
                    <div style='background-color:#1E293B; border-radius: 8px; height: 10px; margin: 15px 0;'>
                        <div style='background: linear-gradient(90deg, #00E5FF 0%, #10B981 100%); width: {row['Probability']}%; height:10px; border-radius:8px;'></div>
                    </div>
                    <span style='font-size:0.9rem; color:#94A3B8;'>Confidence Score: <strong>{row['Confidence']}%</strong></span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Risk details list
            risk_color = "#10B981" if row["Risk_Level"] in ("Very Low", "Low") else ("#F59E0B" if row["Risk_Level"] == "Medium" else "#EF4444")
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h4 style='color:#94A3B8; margin-top:0;'>Risk Assessment</h4>
                    <div style='font-size:1.4rem; font-weight:700; color:{risk_color}; margin-bottom:10px;'>{row['Risk_Level']} Risk</div>
                    <span style='font-size:0.85rem; color:#94A3B8;'>Suggested Sizing: <strong>{row['Position_Size']}</strong></span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_trade:
            # Suggested derivative contract details card
            st.markdown(
                f"""
                <div class='metric-card' style='border-left: 5px solid #00E5FF;'>
                    <h4 style='color:#94A3B8; margin-top:0;'>Suggested Option Position</h4>
                    <div style='font-size: 1.5rem; font-weight: 800; color: #F8FAFC; margin-bottom: 12px;'>
                        Buy {row['Strike']} {row['Option_Type']}
                    </div>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9rem; color:#94A3B8; margin-bottom:15px;'>
                        <div>Target Range: <strong style='color:#F1F5F9;'>₹{row['Target_1']} - ₹{row['Target_3']}</strong></div>
                        <div>Stop Loss: <strong style='color:#EF4444;'>₹{row['Stop_Loss']}</strong></div>
                        <div>Estimated Premium: <strong style='color:#00E5FF;'>{row['Premium_Range']}</strong></div>
                        <div>Contract Expiry: <strong style='color:#F1F5F9;'>{row['Expiry']}</strong></div>
                    </div>
                    <div style='border-top: 1px solid #1E293B; padding-top: 10px; font-size:0.88rem; color:#E2E8F0; line-height:1.4;'>
                        <strong>AI Rationale</strong>:<br>
                        <em>{row['Explanation']}</em>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
