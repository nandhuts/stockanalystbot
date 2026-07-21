import streamlit as st

from ai_stock_advisor.services.llm.chat_assistant import AIChatAssistantClient


def render_chat_assistant() -> None:
    """
    Renders the AI Chat Assistant panel.
    Allows users to query specific stocks (e.g. Reliance) and get data-backed RAG reports.
    """
    st.markdown("<h2 class='sub-title'>AI Chat Advisor</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Chat with our financial advisory agent. Type questions like <em>'Should I buy Reliance?'</em> or ask general market inquiries.</p>",
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am your AI Chat Advisor.** 📈🤖\n\n"
                    "Ask me any question about a specific stock (e.g., *'Should I buy Reliance?'* or *'Analyze INFY.NS'*) "
                    "or ask general queries (e.g., *'What is MACD?'*).\n\n"
                    "If you ask about a stock, I will automatically query live technical indicators, support and resistance levels, "
                    "option chains, and recent news sentiment to compile a comprehensive, data-backed analysis!"
                )
            }
        ]

    # Render conversation log
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input Panel
    user_input = st.chat_input("Ask about Nifty 50 stocks...")

    if user_input:
        # 1. Render user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing message direction..."):
                try:
                    client = AIChatAssistantClient()
                    # Extract ticker
                    ticker = client.extract_ticker(user_input)
                except Exception as exc:
                    st.error(f"Failed connecting to OpenAI API: {exc}")
                    return

            context = None
            if ticker:
                with st.spinner(f"Compiling live data packages, indicators, and news feeds for *{ticker}*..."):
                    try:
                        context = client.compile_context(ticker)
                    except Exception as exc:
                        st.warning(f"Could not load data package for symbol {ticker}: {exc}. Bypassing context.")
                        context = None

            with st.spinner("AI Advisor formulating thesis reasoning..."):
                try:
                    response_text = client.ask_advisor(user_input, context)
                except Exception as exc:
                    response_text = f"❌ Failed formulating advice: {exc}"

            # Render response
            st.markdown(response_text)
            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
