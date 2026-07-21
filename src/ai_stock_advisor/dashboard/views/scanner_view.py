from pathlib import Path
import pandas as pd
import streamlit as st

from config.settings import settings
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.scanner import StockScanner
from ai_stock_advisor.services.market_data.client import MarketDataClient


def render_scanner() -> None:
    """Renders the Market Scanner page."""
    st.markdown("<h2 class='sub-title'>Market Scanner</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Run multi-factor scans over the Nifty 50 stock index to identify bullish entries.</p>",
        unsafe_allow_html=True,
    )

    results_path = Path(settings.BASE_DIR) / "data" / "scan_results.csv"

    # Action panel
    st.markdown("### Run Scan Operations")
    c1, c2 = st.columns([1, 2])
    with c1:
        force_refresh = st.checkbox("Force Refresh", help="Ignore local file cache and download fresh ticker data.")
        run_btn = st.button("🚀 Trigger Market Scan", use_container_width=True)
    with c2:
        st.markdown(
            "<div style='background-color:#161F30; padding:15px; border-radius:8px; border:1px solid #1E293B;'>"
            "<span style='color:#00E5FF; font-weight:600;'>Scan Parameters:</span><br>"
            "<span style='font-size:0.9rem; color:#94A3B8;'>"
            "• Targets: Nifty 50 stocks listed on National Stock Exchange (NSE).<br>"
            "• Filters: EMA 20/50 crossovers, RSI momentum, MACD histograms, Volume surges, ADX trend strength."
            "</span>"
            "</div>",
            unsafe_allow_html=True
        )

    if run_btn:
        with st.spinner("Downloading and processing Nifty 50 stocks..."):
            try:
                # Initialize scanner stack
                client = MarketDataClient()
                engine = TechnicalIndicatorEngine()
                scanner = StockScanner(market_client=client, indicator_engine=engine)
                
                # Execute scan
                df_scan = scanner.scan(force_refresh=force_refresh)
                
                # Execute ranker to generate synchronized rankings_results.csv
                from ai_stock_advisor.core.ranker import StockRanker
                ranker = StockRanker(scanner)
                ranker.rank_stocks(force_refresh=True)
                
                st.success(f"🎉 Scan and AI Rankings completed successfully! Processed {len(df_scan)} tickers.")
            except Exception as exc:
                st.error(f"Scan operation failed: {exc}")

    # Load results
    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
        except Exception as exc:
            st.error(f"Error loading scanner data: {exc}")
            return

        st.markdown("---")
        st.markdown("### Scan Results")
        
        # Filtering controls
        col1, col2 = st.columns([2, 1])
        with col1:
            score_filter = st.slider("Filter by Minimum Score", min_value=0, max_value=100, value=0, step=5)
        with col2:
            search_ticker = st.text_input("Search Ticker Symbol").strip().upper()

        # Apply filters
        filtered_df = df[df["Score"] >= score_filter]
        if search_ticker:
            filtered_df = filtered_df[filtered_df["Ticker"].str.contains(search_ticker)]

        st.markdown(f"Displaying **{len(filtered_df)}** matches:")
        
        # Styled pandas dataframe mapping
        st.dataframe(
            filtered_df,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Trend Score",
                    help="Bullish Momentum Trend Score (0-100)",
                    format="%.0f",
                    min_value=0,
                    max_value=100,
                ),
                "Above_EMA20": "Above EMA20",
                "Above_EMA50": "Above EMA50",
                "EMA_Crossover": "EMA Cross",
                "RSI": st.column_config.NumberColumn("RSI (14)", format="%.1f"),
                "MACD_Bullish": "MACD Bullish",
                "Volume_Spike": "Vol Spike",
                "ADX": st.column_config.NumberColumn("ADX (14)", format="%.1f"),
                "Close": st.column_config.NumberColumn("Close Price", format="₹%.2f"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No scanning logs found. Click the button above to run your first index scan!")
