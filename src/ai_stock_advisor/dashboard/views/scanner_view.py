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

        # ----------------------------------------------------
        # NEW SECTION: AI Machine Learning Models
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("### AI Machine Learning Models")
        st.markdown(
            "<p style='color:#94A3B8;'>Train Random Forest, XGBoost, and LightGBM models on historical index indicators to predict 5-day upward movements.</p>",
            unsafe_allow_html=True
        )
        
        c_ml1, c_ml2 = st.columns([1, 2])
        with c_ml1:
            train_ml_btn = st.button("🤖 Train ML Ensemble Models", use_container_width=True)
            
        cv_report_path = Path(settings.BASE_DIR) / "data" / "models" / "cv_metrics.csv"
        
        if train_ml_btn:
            with st.spinner("Preparing datasets and executing 5-fold cross-validation..."):
                try:
                    from ai_stock_advisor.ml.pipeline import StockMLPipeline
                    pipeline = StockMLPipeline()
                    report = pipeline.train_and_evaluate()
                    st.success("🎉 ML Ensemble models trained and saved successfully!")
                except Exception as exc:
                    st.error(f"ML Pipeline execution failed: {exc}")

        if cv_report_path.exists():
            try:
                cv_df = pd.read_csv(cv_report_path)
                cv_df = cv_df.rename(columns={"Unnamed: 0": "Classifier Model"})
                st.markdown("##### 5-Fold Cross-Validation Metrics Summary:")
                st.dataframe(
                    cv_df,
                    column_config={
                        "Classifier Model": "Classifier Model",
                        "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                        "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                        "f1": st.column_config.NumberColumn("F1-Score", format="%.3f"),
                        "auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as exc:
                st.warning(f"Failed loading CV report file: {exc}")
    else:
        st.info("No scanning logs found. Click the button above to run your first index scan!")
