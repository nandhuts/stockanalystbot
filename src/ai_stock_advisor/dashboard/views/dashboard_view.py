from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from config.settings import settings


def render_dashboard() -> None:
    """Renders the general dashboard summary page."""
    st.markdown("<h2 class='sub-title'>Market Summary & Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;'>Overview of scanned securities, bullish sentiment distribution, and top ratings.</p>", unsafe_allow_html=True)
    
    # Load scan results
    results_path = Path(settings.BASE_DIR) / "data" / "scan_results.csv"
    
    if not results_path.exists():
        st.info("⚠️ No market scan results found. Please navigate to the **Market Scanner** page and trigger a scan to load data.")
        return
        
    try:
        df = pd.read_csv(results_path)
    except Exception as exc:
        st.error(f"Error loading scan results: {exc}")
        return

    # Aggregate stats
    total_scanned = len(df)
    bullish = len(df[df["Score"] >= 70])
    bearish = len(df[df["Score"] <= 30])
    neutral = total_scanned - bullish - bearish
    avg_score = df["Score"].mean()

    # Injected HTML Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div style='color: #94A3B8; font-size: 0.85rem; font-weight: 500;'>TOTAL SECURITIES</div>
                <div style='color: #F1F5F9; font-size: 1.8rem; font-weight: 700; margin-top: 5px;'>{total_scanned}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid #10B981;'>
                <div style='color: #94A3B8; font-size: 0.85rem; font-weight: 500;'>BULLISH (Score ≥ 70)</div>
                <div style='color: #10B981; font-size: 1.8rem; font-weight: 700; margin-top: 5px;'>{bullish}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid #EF4444;'>
                <div style='color: #94A3B8; font-size: 0.85rem; font-weight: 500;'>BEARISH (Score ≤ 30)</div>
                <div style='color: #EF4444; font-size: 1.8rem; font-weight: 700; margin-top: 5px;'>{bearish}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid #00E5FF;'>
                <div style='color: #94A3B8; font-size: 0.85rem; font-weight: 500;'>AVG TREND SCORE</div>
                <div style='color: #00E5FF; font-size: 1.8rem; font-weight: 700; margin-top: 5px;'>{avg_score:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Visuals
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### Sentiment Distribution")
        # Pie Chart
        labels = ["Bullish (Score ≥ 70)", "Neutral (30 < Score < 70)", "Bearish (Score ≤ 30)"]
        values = [bullish, neutral, bearish]
        colors = ["#10B981", "#64748B", "#EF4444"]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.4,
            marker=dict(colors=colors),
            textinfo="percent+label",
            hoverinfo="label+value"
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Top 10 Trending Stocks")
        # Bar Chart
        top10 = df.head(10)
        fig_bar = px.bar(
            top10,
            x="Ticker",
            y="Score",
            color="Score",
            color_continuous_scale=["#64748B", "#00E5FF", "#10B981"],
            range_color=[0, 100],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1E293B", categoryorder="total descending"),
            yaxis=dict(gridcolor="#1E293B", range=[0, 105])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Table Snippet
    st.markdown("##### Leaderboard Overview")
    st.dataframe(
        df[["Ticker", "Close", "Score", "RSI", "ADX"]].head(15),
        use_container_width=True,
        hide_index=True
    )
