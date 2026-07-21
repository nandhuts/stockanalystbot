import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ai_stock_advisor.core.backtester import BacktestingEngine
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient


def render_backtester() -> None:
    """
    Renders the Backtesting View page.
    Simulates a trend-following strategy over 5 years of daily ticker data.
    """
    st.markdown("<h2 class='sub-title'>Strategy Backtester</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Backtest our multi-factor trend-following strategy on 5 years of historical stock data. Evaluates entry triggers, dynamic ATR targets, and stop-losses.</p>",
        unsafe_allow_html=True,
    )

    # Control Configuration
    st.markdown("### Backtest Configurations")
    c_config1, c_config2, c_config3 = st.columns([2, 2, 1])
    
    with c_config1:
        ticker = st.text_input("Enter Stock Ticker Symbol", value="RELIANCE.NS").strip().upper()
    with c_config2:
        initial_cap = st.number_input("Initial Capital (₹)", min_value=10000.0, max_value=10000000.0, value=100000.0, step=50000.0, format="%.2f")
    with c_config3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Backtest", use_container_width=True)

    if run_btn:
        if not ticker:
            st.error("Please provide a valid stock symbol.")
            return

        with st.spinner(f"Downloading 5 years of price history and running simulations for {ticker}..."):
            try:
                # Instantiate backtester stack
                client = MarketDataClient()
                engine = TechnicalIndicatorEngine()
                backtester = BacktestingEngine(client, engine)
                
                results = backtester.run_backtest(ticker, initial_capital=initial_cap)
                st.session_state["bt_results"] = results
                st.success(f"🎉 Backtest completed successfully for {ticker}!")
            except Exception as exc:
                st.error(f"Backtest simulation failed: {exc}")
                return

    # Check session state for loaded backtest results
    if "bt_results" in st.session_state:
        res = st.session_state["bt_results"]
        trades = res["Trades"]
        equity_curve = res["Equity_Curve"]

        # Display Metrics Cards
        st.markdown("---")
        st.markdown(f"### Backtest Performance Summary: *{res['Ticker']}*")
        
        c1, c2, c3, c4 = st.columns(4)
        
        # Color codes
        ret_color = "#10B981" if res["Total_Return_Pct"] >= 0 else "#EF4444"
        win_color = "#10B981" if res["Win_Rate_Pct"] >= 50.0 else "#00E5FF"
        
        with c1:
            st.markdown(
                f"""
                <div class='metric-card' style='text-align:center;'>
                    <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>TOTAL RETURN</div>
                    <div style='font-size:1.8rem; font-weight:700; color:{ret_color}; margin-top:5px;'>{res['Total_Return_Pct']:+.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with c2:
            st.markdown(
                f"""
                <div class='metric-card' style='text-align:center;'>
                    <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>WIN RATE</div>
                    <div style='font-size:1.8rem; font-weight:700; color:{win_color}; margin-top:5px;'>{res['Win_Rate_Pct']}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with c3:
            pf = res["Profit_Factor"]
            pf_color = "#10B981" if pf >= 1.5 else "#00E5FF" if pf >= 1.0 else "#EF4444"
            st.markdown(
                f"""
                <div class='metric-card' style='text-align:center;'>
                    <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>PROFIT FACTOR</div>
                    <div style='font-size:1.8rem; font-weight:700; color:{pf_color}; margin-top:5px;'>{pf}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with c4:
            st.markdown(
                f"""
                <div class='metric-card' style='text-align:center;'>
                    <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>MAX DRAWDOWN</div>
                    <div style='font-size:1.8rem; font-weight:700; color:#EF4444; margin-top:5px;'>{res['Max_Drawdown_Pct']}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Performance summary metrics
        st.markdown(
            f"""
            <div style='margin: 15px 0; background-color:#161F30; padding:15px; border-radius:8px; border:1px solid #1E293B; display:flex; justify-content:space-between; flex-wrap:wrap; font-size:0.92rem; color:#E2E8F0;'>
                <span>• Initial Capital: <strong>₹{res['Initial_Capital']:,.2f}</strong></span>
                <span>• Final Portfolio Value: <strong>₹{res['Final_Capital']:,.2f}</strong></span>
                <span>• Total Trades Executed: <strong>{res['Total_Trades']}</strong></span>
                <span>• Average Gain per Trade: <strong>{res['Average_Return_Pct']:+.2f}%</strong></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Render Equity Curve Chart
        st.markdown("#### Portfolio Equity Curve (5-Year Growth)")
        eq_df = pd.DataFrame(equity_curve)
        eq_df["Date"] = pd.to_datetime(eq_df["Date"])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["Date"],
            y=eq_df["Equity"],
            mode="lines",
            name="Portfolio Equity",
            line=dict(color="#00E5FF", width=2.0),
            fill="tozeroy",
            fillcolor="rgba(0, 229, 255, 0.08)"
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0B0F17",
            font=dict(color="#F1F5F9"),
            margin=dict(t=5, b=5, l=5, r=5),
            height=300,
            xaxis=dict(showgrid=True, gridcolor="#1E293B"),
            yaxis=dict(showgrid=True, gridcolor="#1E293B")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download Report CSV trigger
        st.markdown("#### Executed Trades Log")
        
        if not trades:
            st.info("No trades were opened by the strategy on this ticker history.")
        else:
            trades_df = pd.DataFrame(trades)
            
            # Format columns
            trades_df_display = trades_df.copy()
            trades_df_display = trades_df_display.rename(columns={
                "Buy_Date": "Entry Date",
                "Buy_Price": "Entry Price",
                "Sell_Date": "Exit Date",
                "Sell_Price": "Exit Price",
                "Return_Pct": "Return (%)",
                "Profit": "Profit/Loss (₹)",
                "Exit_Reason": "Exit Reason"
            })
            
            # Show Table
            st.dataframe(
                trades_df_display,
                column_config={
                    "Entry Price": st.column_config.NumberColumn(format="₹%.2f"),
                    "Exit Price": st.column_config.NumberColumn(format="₹%.2f"),
                    "Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Profit/Loss (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )

            # Download CSV Button
            csv_data = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Backtest Trades Report (CSV)",
                data=csv_data,
                file_name=f"backtest_report_{ticker}.csv",
                mime="text/csv",
                use_container_width=True
            )
