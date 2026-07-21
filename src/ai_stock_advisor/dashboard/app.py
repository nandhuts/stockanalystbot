import sys
from pathlib import Path

# Ensure correct module paths are resolved during execution
dashboard_dir = Path(__file__).resolve().parent
src_dir = dashboard_dir.parent.parent
root_dir = src_dir.parent

# Inject paths into sys.path
for path in [str(src_dir), str(root_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st

# Set Streamlit Page Configurations
st.set_page_config(
    page_title="AI Stock Advisor Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import View Panels
from ai_stock_advisor.dashboard.views.dashboard_view import render_dashboard
from ai_stock_advisor.dashboard.views.scanner_view import render_scanner
from ai_stock_advisor.dashboard.views.top_stocks_view import render_top_stocks
from ai_stock_advisor.dashboard.views.charts_view import render_charts
from ai_stock_advisor.dashboard.views.search_view import render_search

# Inject custom premium CSS styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply Custom Font Globally */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hide Streamlit default styling elements for clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Styled Gradient Title Header */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(95deg, #00E5FF 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 25px;
        letter-spacing: -0.5px;
    }
    
    /* Subheadings styling */
    .sub-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    /* Premium Glassmorphic Cards */
    .metric-card {
        background-color: rgba(22, 31, 48, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 229, 255, 0.2);
    }
    
    /* Custom Badges */
    .card-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-right: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation Menu inside Sidebar
st.sidebar.markdown(
    "<h3 style='color: #00E5FF; font-weight:800; font-size:1.4rem; letter-spacing:-0.5px;'>ADVISOR CONSOLE</h3>", 
    unsafe_allow_html=True
)

navigation_menu = {
    "🏠 Dashboard Summary": "dashboard",
    "🔍 Index Market Scanner": "scanner",
    "🏆 Top Bullish Stocks": "top_stocks",
    "📊 Technical Charts": "charts",
    "🔎 Query Custom Symbol": "search",
}

selection = st.sidebar.radio(
    "NAVIGATION", 
    list(navigation_menu.keys()), 
    label_visibility="collapsed"
)

selected_page = navigation_menu[selection]

# Header Bar
st.markdown("<h1 class='main-title'>AI Stock Advisor Terminal</h1>", unsafe_allow_html=True)

# Render Target Panels
if selected_page == "dashboard":
    render_dashboard()
elif selected_page == "scanner":
    render_scanner()
elif selected_page == "top_stocks":
    render_top_stocks()
elif selected_page == "charts":
    render_charts()
elif selected_page == "search":
    render_search()
