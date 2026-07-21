from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient
from ai_stock_advisor.services.market_data.constants import NIFTY_50_TICKERS


def render_charts() -> None:
    """Renders the interactive Charts page with financial subplots."""
    st.markdown("<h2 class='sub-title'>Technical Charts</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Interactive technical analysis charts featuring price action overlays and oscillators.</p>",
        unsafe_allow_html=True,
    )

    # Inputs
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        ticker = st.selectbox("Select Securities", list(NIFTY_50_TICKERS))
    with c2:
        period = st.selectbox("Historical Period", ["3mo", "6mo", "1y", "2y"], index=2)
    with c3:
        interval = st.selectbox("Timeframe Interval", ["1d", "1wk"], index=0)

    # Initialize Clients
    client = MarketDataClient()
    engine = TechnicalIndicatorEngine()

    with st.spinner(f"Loading data for {ticker}..."):
        try:
            df = client.fetch_ohlcv(ticker, period=period, interval=interval)
            # Calculate all indicators
            df_ind = engine.compute_all_indicators(df)
        except Exception as exc:
            st.error(f"Error loading charts: {exc}")
            return

    # Check length
    if len(df_ind) < 20:
        st.warning("Insufficient data points to render technical indicators.")
        return

    # Let's split charts into two distinct sections: Price/Volume Panel and Oscillator Panel

    # 1. PRICE & VOLUME PANEL
    st.markdown("### Price Action & Volume")
    
    # Create subplots: Row 1 = Candlesticks + EMAs + Bollinger, Row 2 = Volume + VWAP
    fig_price = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06, 
        row_heights=[0.75, 0.25]
    )

    # Candlestick
    fig_price.add_trace(
        go.Candlestick(
            x=df_ind.index,
            open=df_ind["Open"],
            high=df_ind["High"],
            low=df_ind["Low"],
            close=df_ind["Close"],
            name="Price",
            increasing_line_color="#10B981",  # Green
            decreasing_line_color="#EF4444",  # Red
        ),
        row=1, col=1
    )

    # EMA Overlays
    fig_price.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA_20"], name="EMA 20", line=dict(color="#00E5FF", width=1.5)), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA_50"], name="EMA 50", line=dict(color="#F59E0B", width=1.5)), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA_200"], name="EMA 200", line=dict(color="#EC4899", width=1.5)), row=1, col=1)

    # Bollinger Bands
    fig_price.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"], name="BB Upper", line=dict(color="#64748B", width=1, dash="dash")), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"], name="BB Lower", line=dict(color="#64748B", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(100, 116, 139, 0.05)"), row=1, col=1)

    # Volume & VWAP
    fig_price.add_trace(
        go.Bar(
            x=df_ind.index, 
            y=df_ind["Volume"], 
            name="Volume", 
            marker_color="rgba(100, 116, 139, 0.4)"
        ),
        row=2, col=1
    )
    fig_price.add_trace(
        go.Scatter(
            x=df_ind.index, 
            y=df_ind["VWAP"], 
            name="VWAP", 
            line=dict(color="#3B82F6", width=1.5)
        ),
        row=1, col=1
    )

    # Layout Styling
    fig_price.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F17",
        font=dict(color="#F1F5F9"),
        xaxis_rangeslider_visible=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_price.update_xaxes(gridcolor="#1E293B", showline=True, linecolor="#1E293B")
    fig_price.update_yaxes(gridcolor="#1E293B", showline=True, linecolor="#1E293B")

    st.plotly_chart(fig_price, use_container_width=True)

    # 2. OSCILLATORS PANEL (RSI, MACD, ADX)
    st.markdown("### Oscillators & Momentum")

    fig_osc = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.33]
    )

    # RSI
    fig_osc.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI_14"], name="RSI", line=dict(color="#8B5CF6", width=2)), row=1, col=1)
    # Dotted oversold/overbought lines
    fig_osc.add_shape(type="line", x0=df_ind.index[0], x1=df_ind.index[-1], y0=70, y1=70, line=dict(color="#EF4444", width=1, dash="dash"), row=1, col=1)
    fig_osc.add_shape(type="line", x0=df_ind.index[0], x1=df_ind.index[-1], y0=30, y1=30, line=dict(color="#10B981", width=1, dash="dash"), row=1, col=1)

    # MACD
    fig_osc.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"], name="MACD", line=dict(color="#3B82F6", width=1.5)), row=2, col=1)
    fig_osc.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_Signal"], name="Signal", line=dict(color="#EF4444", width=1.5)), row=2, col=1)
    # Histogram as color bars
    colors = ["#10B981" if val >= 0 else "#EF4444" for val in df_ind["MACD_Hist"]]
    fig_osc.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_Hist"], name="Histogram", marker_color=colors), row=2, col=1)

    # ADX
    fig_osc.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ADX_14"], name="ADX", line=dict(color="#E2E8F0", width=2)), row=3, col=1)
    fig_osc.add_shape(type="line", x0=df_ind.index[0], x1=df_ind.index[-1], y0=25, y1=25, line=dict(color="#F59E0B", width=1, dash="dash"), row=3, col=1)

    fig_osc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B0F17",
        font=dict(color="#F1F5F9"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_osc.update_xaxes(gridcolor="#1E293B", showline=True, linecolor="#1E293B")
    fig_osc.update_yaxes(gridcolor="#1E293B", showline=True, linecolor="#1E293B")

    st.plotly_chart(fig_osc, use_container_width=True)
