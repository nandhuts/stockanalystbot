from pathlib import Path
import pandas as pd
import streamlit as st
from config.settings import settings


def render_top_stocks() -> None:
    """
    Renders the Top Stocks page showing the Top 20 stocks
    ranked by bullish breakout probability scores.
    """
    st.markdown("<h2 class='sub-title'>AI Stock Rankings (Top 20)</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Top 20 securities ranked by their bullish trend probability scores, combining Trend, Momentum, and Volume factors.</p>",
        unsafe_allow_html=True,
    )

    results_path = Path(settings.BASE_DIR) / "data" / "rankings_results.csv"

    # If rankings don't exist, try loading scanner data and running the ranker on it
    if not results_path.exists():
        st.info("⚠️ Rankings data not found. Please navigate to the **Market Scanner** page and trigger a scan first to generate rankings.")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as exc:
        st.error(f"Error loading stock rankings: {exc}")
        return

    if df.empty:
        st.warning("The rankings list is currently empty. Run a scan to populate results.")
        return

    st.markdown(f"Displaying the top **{len(df)}** securities sorted by bullish trend probability:")
    st.markdown("<br>", unsafe_allow_html=True)

    # Render a responsive card grid for the Top 20 stocks
    cols = st.columns(3)
    
    for idx, (_, row) in enumerate(df.iterrows()):
        col = cols[idx % 3]
        
        ticker = row["Ticker"]
        prob_score = float(row["Probability_Score"])
        close = float(row["Close"])
        rsi = float(row["RSI"])
        adx = float(row["ADX"])
        
        # Color rating based on probability
        if prob_score >= 75.0:
            color = "#10B981"  # Emerald Green
            bg_color = "rgba(16, 185, 129, 0.15)"
        elif prob_score >= 50.0:
            color = "#00E5FF"  # Neon Cyan
            bg_color = "rgba(0, 229, 255, 0.15)"
        else:
            color = "#EF4444"  # Rose Red
            bg_color = "rgba(239, 68, 68, 0.15)"

        # Badges representing triggers
        badges = []
        if row["Above_EMA20"]:
            badges.append("<span class='card-badge' style='background-color:rgba(0,229,255,0.12); color:#00E5FF;'>EMA20</span>")
        if row["EMA_Crossover"]:
            badges.append("<span class='card-badge' style='background-color:rgba(16,185,129,0.12); color:#10B981;'>EMA CROSS</span>")
        if row["Volume_Spike"]:
            badges.append("<span class='card-badge' style='background-color:rgba(245,158,11,0.12); color:#F59E0B;'>VOL SPIKE</span>")
            
        badges_html = " ".join(badges) if badges else "<span style='color:#64748B; font-size:0.8rem;'>No active triggers</span>"

        col.markdown(
            f"""
            <div class='metric-card' style='border-top: 4px solid {color}; min-height: 220px; display: flex; flex-direction: column; justify-content: space-between;'>
                <div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-size: 1.15rem; font-weight: 700; color:#F8FAFC;'>#{idx + 1} {ticker}</span>
                        <span style='background-color: {bg_color}; color: {color}; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight: 700;'>
                            {prob_score}% Prob
                        </span>
                    </div>
                    <div style='margin-top: 10px; font-size: 1.4rem; font-weight: 700; color:#00E5FF;'>
                        ₹{close:,.2f}
                    </div>
                    <div style='margin-top: 12px; font-size: 0.85rem; color:#94A3B8;'>
                        RSI: <strong style='color:#F1F5F9;'>{rsi:.1f}</strong> | ADX: <strong style='color:#F1F5F9;'>{adx:.1f}</strong>
                    </div>
                </div>
                <div style='margin-top: 15px; border-top: 1px solid #1E293B; padding-top: 10px;'>
                    {badges_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
