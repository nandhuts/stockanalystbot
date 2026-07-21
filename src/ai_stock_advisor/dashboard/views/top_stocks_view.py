from pathlib import Path
import pandas as pd
import streamlit as st
from config.settings import settings


def render_top_stocks() -> None:
    """Renders the Top Stocks page with a card grid of highly rated stocks."""
    st.markdown("<h2 class='sub-title'>Top Bullish Signals</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Securities exhibiting the strongest bullish alignment (Score ≥ 70).</p>",
        unsafe_allow_html=True,
    )

    results_path = Path(settings.BASE_DIR) / "data" / "scan_results.csv"

    if not results_path.exists():
        st.info("⚠️ Scan results not found. Run a scan from the **Market Scanner** page first.")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as exc:
        st.error(f"Error loading scan results: {exc}")
        return

    # Filter bullish stocks
    top_df = df[df["Score"] >= 70]
    
    if top_df.empty:
        st.markdown(
            "<div style='background-color:#161F30; padding:20px; border-radius:8px; border:1px solid #1E293B; text-align:center;'>"
            "<h5>No stocks met the strict bullish criterion (Score ≥ 70) in the latest scan.</h5>"
            "<p style='color:#94A3B8; margin-top:10px;'>Consider lowering your criteria or executing a refresh on the scanner.</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>##### Stocks with Moderate Bullish Momentum (Score ≥ 50):", unsafe_allow_html=True)
        top_df = df[df["Score"] >= 50]
        if top_df.empty:
            st.info("No stocks with moderate bullish indicators found.")
            return

    # Render a responsive card grid
    # We display up to 9 cards in a 3-column layout
    max_cards = 12
    display_df = top_df.head(max_cards)
    
    cols = st.columns(3)
    
    for idx, (_, row) in enumerate(display_df.iterrows()):
        col = cols[idx % 3]
        
        ticker = row["Ticker"]
        score = int(row["Score"])
        close = float(row["Close"])
        rsi = float(row["RSI"])
        adx = float(row["ADX"])
        
        # Build trigger badges
        triggers = []
        if row["Above_EMA20"]:
            triggers.append("<span class='card-badge' style='background-color:rgba(0,229,255,0.15); color:#00E5FF;'>EMA20</span>")
        if row["EMA_Crossover"]:
            triggers.append("<span class='card-badge' style='background-color:rgba(16,185,129,0.15); color:#10B981;'>Bull Cross</span>")
        if row["Volume_Spike"]:
            triggers.append("<span class='card-badge' style='background-color:rgba(245,158,11,0.15); color:#F59E0B;'>Volume Surge</span>")
            
        triggers_html = " ".join(triggers) if triggers else "<span style='color:#64748B;'>No trigger badges</span>"

        # Card body with glassmorphic styles
        col.markdown(
            f"""
            <div class='metric-card' style='border-top: 4px solid #10B981; min-height: 220px; display: flex; flex-direction: column; justify-content: space-between;'>
                <div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-size: 1.15rem; font-weight: 700; color:#F8FAFC;'>{ticker}</span>
                        <span style='background-color: rgba(16,185,129,0.2); color:#10B981; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;'>
                            {score} pts
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
                    {triggers_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
