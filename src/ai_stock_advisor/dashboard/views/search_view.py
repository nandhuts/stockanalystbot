import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config.settings import settings
from ai_stock_advisor.services.market_data.client import MarketDataClient
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.scanner import StockScanner


def render_search() -> None:
    """Renders the Search Stock page to query custom ticker analyses."""
    st.markdown("<h2 class='sub-title'>Search Custom Security</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Input any Yahoo Finance symbol to evaluate trend signals and compute its custom rating score.</p>",
        unsafe_allow_html=True,
    )

    # Search inputs
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_symbol = st.text_input(
            "Enter Stock Symbol", 
            value="AAPL", 
            help="For Indian stocks use the suffix .NS (e.g. RELIANCE.NS, TCS.NS). For US stocks use standard ticker symbols (e.g. AAPL, MSFT)."
        ).strip().upper()
    with col2:
        period = st.selectbox("Lookback Period", ["6mo", "1y", "2y"], index=1)

    if custom_symbol:
        client = MarketDataClient()
        engine = TechnicalIndicatorEngine()
        scanner = StockScanner(market_client=client, indicator_engine=engine)
        
        with st.spinner(f"Analyzing {custom_symbol}..."):
            try:
                # Fetch price history (force refresh on search to guarantee live data)
                df = client.fetch_ohlcv(custom_symbol, period=period, interval="1d", force_refresh=True)
                
                # Check data length
                if len(df) < 50:
                    st.warning(f"Ticker '{custom_symbol}' returned insufficient historical rows ({len(df)}) for analysis.")
                    return
                
                # Compute scores and indicators
                score_dict = scanner.score_stock(df)
                df_ind = engine.compute_all_indicators(df)
                latest = df_ind.iloc[-1]
                
            except Exception as exc:
                st.error(f"Analysis failed for ticker '{custom_symbol}': {exc}")
                return

        # Render Score Badge and Metrics
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            score = int(score_dict["Score"])
            if score >= 70:
                border_color = "#10B981"
                status_text = "Strong Bullish"
            elif score >= 50:
                border_color = "#00E5FF"
                status_text = "Moderate Bullish"
            else:
                border_color = "#EF4444"
                status_text = "Bearish / Weak Neutral"

            st.markdown(
                f"""
                <div class='metric-card' style='border-top: 5px solid {border_color}; text-align:center; min-height: 200px; display:flex; flex-direction:column; justify-content:center;'>
                    <div style='font-size:0.9rem; color:#94A3B8; font-weight:500;'>RATING SCORE</div>
                    <div style='font-size:3.5rem; font-weight:800; color:{border_color}; margin: 10px 0;'>{score}</div>
                    <div style='font-size:0.95rem; font-weight:600; color:#F1F5F9;'>{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with c2:
            st.markdown(f"#### {custom_symbol} Technical Profile")
            st.markdown(f"**Latest Close Price: ₹{score_dict['Close']:.2f}**")
            
            # Bullets of indicator triggers
            cols_grid = st.columns(2)
            with cols_grid[0]:
                st.markdown(f"• Above EMA 20: {'✅ **Yes**' if score_dict['Above_EMA20'] else '❌ **No**'}")
                st.markdown(f"• Above EMA 50: {'✅ **Yes**' if score_dict['Above_EMA50'] else '❌ **No**'}")
                st.markdown(f"• EMA Crossover (20 > 50): {'✅ **Yes**' if score_dict['EMA_Crossover'] else '❌ **No**'}")
            with cols_grid[1]:
                st.markdown(f"• RSI (14): **{score_dict['RSI']:.1f}** ({'Bullish' if score_dict['RSI'] > 50 else 'Bearish/Neutral'})")
                st.markdown(f"• MACD Bullish: {'✅ **Yes**' if score_dict['MACD_Bullish'] else '❌ **No**'}")
                st.markdown(f"• Volume Surge (1.5x MA): {'✅ **Yes**' if score_dict['Volume_Spike'] else '❌ **No**'}")
                st.markdown(f"• ADX (14) Trend: **{score_dict['ADX']:.1f}** ({'Strong Trend' if score_dict['ADX'] > 25 else 'Range-bound'})")

        # Rendering Plotly Candlestick Chart
        st.markdown("<br>##### Price Chart", unsafe_allow_html=True)
        fig = go.Figure()
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_ind.index,
            open=df_ind["Open"],
            high=df_ind["High"],
            low=df_ind["Low"],
            close=df_ind["Close"],
            name="Candlestick",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444"
        ))
        # Indicators
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA_20"], name="EMA 20", line=dict(color="#00E5FF", width=1.5)))
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA_50"], name="EMA 50", line=dict(color="#F59E0B", width=1.5)))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0B0F17",
            font=dict(color="#F1F5F9"),
            xaxis_rangeslider_visible=False,
            margin=dict(t=5, b=5, l=5, r=5),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(gridcolor="#1E293B", showline=True, linecolor="#1E293B")
        fig.update_yaxes(gridcolor="#1E293B", showline=True, linecolor="#1E293B")
        
        st.plotly_chart(fig, use_container_width=True)
