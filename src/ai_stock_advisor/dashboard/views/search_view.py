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

        # ----------------------------------------------------
        # NEW SECTION: Option Chain Analysis
        # ----------------------------------------------------
        from ai_stock_advisor.core.options import OptionAnalyzer
        opt_analyzer = OptionAnalyzer(client, engine)
        
        has_options = False
        try:
            opt_report = opt_analyzer.analyze_options(custom_symbol)
            has_options = True
        except Exception as exc:
            has_options = False

        if has_options and opt_report:
            st.markdown("<br>##### Option Chain Sentiment (Nearest Expiry)", unsafe_allow_html=True)
            
            # Sub-panel layout: metric cards
            c_opt1, c_opt2, c_opt3, c_opt4 = st.columns(4)
            
            with c_opt1:
                pcr_val = opt_report["PCR"]
                pcr_color = "#10B981" if pcr_val >= 1.25 else "#EF4444" if pcr_val <= 0.75 else "#00E5FF"
                st.markdown(
                    f"""
                    <div class='metric-card' style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>PUT-CALL RATIO (PCR)</div>
                        <div style='font-size:1.6rem; font-weight:700; color:{pcr_color}; margin-top:5px;'>{pcr_val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with c_opt2:
                max_pain_val = opt_report["Max_Pain"]
                st.markdown(
                    f"""
                    <div class='metric-card' style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>MAX PAIN POINT</div>
                        <div style='font-size:1.6rem; font-weight:700; color:#F1F5F9; margin-top:5px;'>₹{max_pain_val:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with c_opt3:
                atm_strike_val = opt_report["ATM_Strike"]
                st.markdown(
                    f"""
                    <div class='metric-card' style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>ATM STRIKE PRICE</div>
                        <div style='font-size:1.6rem; font-weight:700; color:#F59E0B; margin-top:5px;'>₹{atm_strike_val:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with c_opt4:
                opt_sentiment = opt_report["Sentiment"]
                opt_color = "#10B981" if opt_sentiment == "BULLISH" else "#EF4444" if opt_sentiment == "BEARISH" else "#64748B"
                st.markdown(
                    f"""
                    <div class='metric-card' style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#94A3B8; font-weight:500;'>OPTION SENTIMENT</div>
                        <div style='font-size:1.55rem; font-weight:700; color:{opt_color}; margin-top:5px;'>{opt_sentiment}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Recommendation Card
            st.markdown(
                f"""
                <div class='metric-card' style='background-color:#161F30; border:1px solid #1E293B;'>
                    <div style='font-weight:700; color:#00E5FF; font-size:1rem; margin-bottom:10px;'>💡 Volatility-Adjusted Trading Suggestion:</div>
                    <div style='display:flex; justify-content:space-between; flex-wrap:wrap;'>
                        <span style='color:#F8FAFC; font-size:0.92rem;'>• Action: <strong>{opt_report['Suggestion']} (Strike: {opt_report['Suggested_Strike']:.1f})</strong></span>
                        <span style='color:#10B981; font-size:0.92rem;'>• Target Price: <strong>₹{opt_report['Target']:.2f}</strong></span>
                        <span style='color:#EF4444; font-size:0.92rem;'>• Stop Loss Price: <strong>₹{opt_report['Stop_Loss']:.2f}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # AI Option Advisor Suggestion Block
            from ai_stock_advisor.services.llm.option_advisor import AIOptionAdvisorClient, OptionSentimentEnum
            
            advisor_client = AIOptionAdvisorClient()
            with st.spinner("AI Option Advisor analyzing parameters..."):
                try:
                    advice = advisor_client.generate_option_trade(custom_symbol, score_dict, opt_report)
                except Exception as exc:
                    st.warning(f"Failed loading AI Options advice: {exc}")
                    advice = None

            if advice:
                sentiment_label = advice.sentiment.value
                sentiment_color = "#10B981" if advice.sentiment == OptionSentimentEnum.BULLISH else "#EF4444" if advice.sentiment == OptionSentimentEnum.BEARISH else "#64748B"
                bg_color = "rgba(16, 185, 129, 0.12)" if advice.sentiment == OptionSentimentEnum.BULLISH else "rgba(239, 68, 68, 0.12)" if advice.sentiment == OptionSentimentEnum.BEARISH else "rgba(100, 116, 139, 0.12)"
                
                st.markdown(
                    f"""
                    <div class='metric-card' style='border-top: 4px solid #00E5FF; padding: 20px; margin-top: 15px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 15px;'>
                            <span style='font-size:1.1rem; font-weight:700; color:#F8FAFC;'>🤖 AI Option Advisor Analysis</span>
                            <span style='background-color:{bg_color}; color:{sentiment_color}; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;'>
                                Probability: {advice.probability_score}% | {sentiment_label}
                            </span>
                        </div>
                        <div style='margin-bottom:15px; background-color:#0B0F17; padding:12px; border-radius:6px; border:1px solid #1E293B;'>
                            <div style='font-size:0.82rem; color:#94A3B8; font-weight:500; text-transform:uppercase;'>Recommended Action</div>
                            <div style='font-size:1.25rem; font-weight:700; color:#00E5FF; margin-top:4px;'>
                                {advice.suggested_strategy} (Strike: ₹{advice.strike_price:,.2f})
                            </div>
                            <div style='display:flex; gap:20px; margin-top:10px; font-size:0.88rem;'>
                                <span style='color:#10B981;'>Target: <strong>₹{advice.target_price:,.2f}</strong></span>
                                <span style='color:#EF4444;'>Stop Loss: <strong>₹{advice.stop_loss:,.2f}</strong></span>
                            </div>
                        </div>
                        <div style='font-size:0.9rem; line-height:1.6; color:#CBD5E1;'>
                            <strong>Thesis & Reasoning:</strong><br>
                            <em>{advice.thesis_reasoning}</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # NEW SECTION: AI News Sentiment Analyzer
        # ----------------------------------------------------
        st.markdown("<br>##### Recent News Sentiment Analysis (AI Consolidated)", unsafe_allow_html=True)
        
        from ai_stock_advisor.services.llm.news_analyzer import NewsAnalyzer, SentimentEnum
        
        news_analyzer = NewsAnalyzer()
        with st.spinner("Analyzing recent news headlines..."):
            try:
                report = news_analyzer.analyze_news(custom_symbol, max_articles=5)
            except Exception as exc:
                st.warning(f"Failed analyzing ticker news headlines: {exc}")
                report = None

        if report:
            # Overall News Score Card
            avg_score = report.average_sentiment_score
            overall_sentiment = report.overall_sentiment.value
            
            if report.overall_sentiment == SentimentEnum.POSITIVE:
                color = "#10B981"
                bg_color = "rgba(16, 185, 129, 0.15)"
            elif report.overall_sentiment == SentimentEnum.NEGATIVE:
                color = "#EF4444"
                bg_color = "rgba(239, 68, 68, 0.15)"
            else:
                color = "#64748B"
                bg_color = "rgba(100, 116, 139, 0.15)"
                
            st.markdown(
                f"""
                <div class='metric-card' style='display:flex; align-items:center; justify-content:space-between; border-left: 5px solid {color}; padding: 15px 20px;'>
                    <div>
                        <div style='font-size:0.82rem; color:#94A3B8; font-weight:500;'>CONSOLIDATED SENTIMENT</div>
                        <div style='font-size:1.35rem; font-weight:700; color:{color}; margin-top:5px;'>{overall_sentiment}</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:0.82rem; color:#94A3B8; font-weight:500;'>POLARITY SCORE</div>
                        <span style='background-color:{bg_color}; color:{color}; padding:4px 10px; border-radius:12px; font-weight:700; display:inline-block; margin-top:5px; font-size:1.15rem;'>
                            {avg_score:+.2f}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display individual articles in expanders
            if not report.articles:
                st.info("No news stories were parsed for this security.")
            else:
                for idx, article in enumerate(report.articles, 1):
                    art_sentiment = article.sentiment.value
                    art_score = article.sentiment_score
                    
                    if article.sentiment == SentimentEnum.POSITIVE:
                        badge_color = "rgba(16, 185, 129, 0.15)"
                        text_color = "#10B981"
                    elif article.sentiment == SentimentEnum.NEGATIVE:
                        badge_color = "rgba(239, 68, 68, 0.15)"
                        text_color = "#EF4444"
                    else:
                        badge_color = "rgba(100, 116, 139, 0.15)"
                        text_color = "#94A3B8"

                    with st.expander(f"📰 {article.headline}"):
                        st.markdown(
                            f"""
                            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                                <span class='card-badge' style='background-color:{badge_color}; color:{text_color}; font-size:0.75rem; font-weight:700;'>
                                    {art_sentiment}
                                </span>
                                <span style='font-size:0.85rem; color:#94A3B8; font-weight:600;'>
                                    Polarity: <strong style='color:{text_color};'>{art_score:+.2f}</strong>
                                </span>
                            </div>
                            <p style='color:#E2E8F0; font-size:0.92rem; line-height:1.5; margin:0;'>
                                <strong>AI Summary:</strong> <em>{article.summary}</em>
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.info("Could not fetch news sentiment for this stock ticker.")
