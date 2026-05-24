"""Custom CSS for a polished dark-mode dashboard."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0F0F13;
    --bg-secondary: #1A1A2E;
    --bg-card: #16162A;
    --bg-card-hover: #1E1E3A;
    --accent: #7C3AED;
    --accent-light: #A78BFA;
    --success: #10B981;
    --danger: #EF4444;
    --warning: #F59E0B;
    --text-primary: #E4E4E7;
    --text-secondary: #A1A1AA;
    --border: #27273F;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

/* Headers */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

h1 {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent);
    transform: translateY(-1px);
}

[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.75rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #6D28D9 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
}

/* Form inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stDateInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* Toast/alerts */
.stAlert {
    border-radius: 8px !important;
    border: none !important;
}

/* Plotly charts dark bg */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Selectbox styling */
[data-baseweb="select"] {
    background: var(--bg-card) !important;
}

/* Tag pills */
.tag-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

.tag-smoke {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.tag-real {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.tag-live {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.tag-pending {
    background: rgba(124, 58, 237, 0.15);
    color: #A78BFA;
    border: 1px solid rgba(124, 58, 237, 0.3);
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* Card container helper */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
</style>
"""
