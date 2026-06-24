from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Optional

APP_BRAND_FULL = "Financial Analysis and Stock Trading Analysis"
APP_BRAND_TAGLINE = "Real-time market insights"

NAV_DEFINITION = [
    ("Dashboard", "✨", "Dashboard"),
    ("AI Analyst", "🤖", "Agentic AI"),
    ("About the Project", "🏠", "Overview"),
    ("Live News Sentiment", "📰", "News & sentiment"),
    ("Company Basic Details", "📋", "Company profile"),
    ("Company Advanced Details", "📊", "Technicals"),
    ("Google Trends with Forecast", "🔎", "Search trends"),
    ("Twitter Trends", "𝕏", "Social trends"),
    ("Meeting Summarization", "🎙️", "Meeting notes"),
    ("Stock Future Prediction", "🔮", "Forecast"),
]


def _nav_display(internal, icon, short):
    return f"{icon}  {short}"


def init_theme_state(st):
    if "ui_theme_is_light" not in st.session_state:
        st.session_state["ui_theme_is_light"] = True

    if "global_ticker_search" not in st.session_state:
        st.session_state["global_ticker_search"] = "AAPL"


def _theme_palette(is_light: bool) -> dict[str, str]:
    if is_light:
        return {
            "bg": "#F4F7FB",
            "bg_soft": "#EEF2F7",
            "panel": "#FFFFFF",
            "sidebar": "#FFFFFF",
            "text": "#1E293B",
            "text_strong": "#0F172A",
            "muted": "#64748B",
            "border": "rgba(30, 41, 59, 0.10)",
            "shadow": "0 4px 20px rgba(15, 23, 42, 0.06)",
            "shadow_lg": "0 12px 32px rgba(15, 23, 42, 0.08)",
            "accent": "#0F766E",
            "accent_soft": "rgba(15, 118, 110, 0.10)",
            "accent_2": "#0369A1",
            "accent_2_soft": "rgba(3, 105, 161, 0.10)",
            "input_bg": "#FFFFFF",
            "input_text": "#0F172A",
            "input_placeholder": "#94A3B8",
            "select_bg": "#FFFFFF",
            "hover_bg": "rgba(15, 118, 110, 0.06)",
            "card_bg": "#FFFFFF",
            "card_bg_alt": "linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%)",
            "topbar_bg": "linear-gradient(135deg, rgba(15,118,110,0.08), rgba(3,105,161,0.05)), #FFFFFF",
            "sidebar_brand_bg": "linear-gradient(135deg, rgba(15,118,110,0.08), rgba(3,105,161,0.05))",
            "btn_bg": "#0F766E",
            "btn_bg_hover": "#115E59",
            "btn_text": "#FFFFFF",
            "btn_border": "#0F766E",
            "btn_secondary_bg": "#FFFFFF",
            "btn_secondary_text": "#0F172A",
            "btn_secondary_border": "#CBD5E1",
            "btn_secondary_hover": "#F8FAFC",
            "radio_hover": "rgba(15, 118, 110, 0.08)",
            "gradient_glow_1": "rgba(15, 118, 110, 0.10)",
            "gradient_glow_2": "rgba(3, 105, 161, 0.08)",
        }
    return {
        "bg": "#131A24",
        "bg_soft": "#1A2332",
        "panel": "#1E293B",
        "sidebar": "#1A2332",
        "text": "#E8EEF5",
        "text_strong": "#F8FAFC",
        "muted": "#94A3B8",
        "border": "rgba(148, 163, 184, 0.18)",
        "shadow": "0 4px 20px rgba(0, 0, 0, 0.25)",
        "shadow_lg": "0 14px 36px rgba(0, 0, 0, 0.32)",
        "accent": "#2DD4BF",
        "accent_soft": "rgba(45, 212, 191, 0.14)",
        "accent_2": "#38BDF8",
        "accent_2_soft": "rgba(56, 189, 248, 0.10)",
        "input_bg": "#FFFFFF",
        "input_text": "#0F172A",
        "input_placeholder": "#64748B",
        "select_bg": "#FFFFFF",
        "hover_bg": "rgba(45, 212, 191, 0.10)",
        "card_bg": "rgba(255, 255, 255, 0.03)",
        "card_bg_alt": "linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))",
        "topbar_bg": "linear-gradient(135deg, rgba(45,212,191,0.10), rgba(56,189,248,0.08)), rgba(26,35,50,0.95)",
        "sidebar_brand_bg": "linear-gradient(135deg, rgba(45,212,191,0.10), rgba(56,189,248,0.08))",
        "btn_bg": "#0D9488",
        "btn_bg_hover": "#14B8A6",
        "btn_text": "#FFFFFF",
        "btn_border": "#14B8A6",
        "btn_secondary_bg": "#1E293B",
        "btn_secondary_text": "#F8FAFC",
        "btn_secondary_border": "rgba(148, 163, 184, 0.35)",
        "btn_secondary_hover": "#243044",
        "radio_hover": "rgba(255, 255, 255, 0.06)",
        "gradient_glow_1": "rgba(45, 212, 191, 0.12)",
        "gradient_glow_2": "rgba(56, 189, 248, 0.10)",
    }


def inject_global_css(st):
    is_light = st.session_state.get("ui_theme_is_light", True)
    c = _theme_palette(is_light)

    pos = "#16A34A" if is_light else "#4ADE80"
    neg = "#DC2626" if is_light else "#F87171"
    warn = "#D97706" if is_light else "#FBBF24"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', system-ui, sans-serif !important;
            color: {c["text"]};
        }}

        .stApp {{
            background:
                radial-gradient(circle at 8% 0%, {c["gradient_glow_1"]}, transparent 34%),
                radial-gradient(circle at 92% 4%, {c["gradient_glow_2"]}, transparent 30%),
                linear-gradient(180deg, {c["bg"]} 0%, {c["bg_soft"]} 100%) !important;
        }}

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        section.main > div,
        [data-testid="stAppViewContainer"] > .main {{
            background: transparent !important;
            background-image: none !important;
        }}

        [data-testid="collapsedControl"] {{
            color: {c["text_strong"]} !important;
        }}

        .block-container {{
            max-width: 1320px;
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
        }}

        h1, h2, h3 {{
            letter-spacing: -0.03em;
            font-weight: 700 !important;
            color: {c["text_strong"]} !important;
        }}

        p, label, .stMarkdown, .stCaption {{
            color: {c["muted"]};
        }}

        section[data-testid="stSidebar"] > div {{
            background: {c["sidebar"]} !important;
            border-right: 1px solid {c["border"]} !important;
            box-shadow: {c["shadow"]};
        }}

        section[data-testid="stSidebar"] * {{
            color: {c["text"]} !important;
        }}

        section[data-testid="stSidebar"] .stCaption {{
            color: {c["muted"]} !important;
        }}

        /* ── Form controls: always readable typed text (esp. dark mode) ── */
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTextArea textarea,
        textarea,
        [data-testid="stChatInput"] textarea,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="input"] input {{
            background-color: {c["input_bg"]} !important;
            color: {c["input_text"]} !important;
            -webkit-text-fill-color: {c["input_text"]} !important;
            caret-color: {c["input_text"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 12px !important;
        }}

        .stTextInput input::placeholder,
        .stNumberInput input::placeholder,
        textarea::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {c["input_placeholder"]} !important;
            -webkit-text-fill-color: {c["input_placeholder"]} !important;
            opacity: 1 !important;
        }}

        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stTextArea label,
        .stSlider label,
        .stCheckbox label,
        .stToggle label {{
            color: {c["text"]} !important;
            font-weight: 500 !important;
        }}

        .stTextInput input:focus,
        .stNumberInput input:focus,
        textarea:focus,
        [data-testid="stChatInput"] textarea:focus {{
            border-color: {c["accent"]} !important;
            box-shadow: 0 0 0 3px {c["accent_soft"]} !important;
            outline: none !important;
        }}

        /* Selectbox */
        .stSelectbox div[data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="select"] {{
            background-color: {c["select_bg"]} !important;
            border-radius: 12px !important;
            border-color: {c["border"]} !important;
        }}

        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] div {{
            color: {c["input_text"]} !important;
        }}

        /* Sidebar inputs inherit readable colors too */
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] textarea {{
            background-color: {c["input_bg"]} !important;
            color: {c["input_text"]} !important;
            -webkit-text-fill-color: {c["input_text"]} !important;
        }}

        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {{
            color: {c["input_text"]} !important;
        }}

        /* ── Navigation radio (sidebar) ── */
        .stRadio > div {{
            gap: 0.3rem;
        }}

        .stRadio label {{
            border-radius: 12px !important;
            padding: 0.6rem 0.8rem !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            transition: all 0.2s ease;
            color: {c["text"]} !important;
            font-weight: 500 !important;
        }}

        .stRadio label:hover {{
            background: {c["radio_hover"]} !important;
            border-color: {c["border"]} !important;
        }}

        .stRadio label[data-checked="true"],
        .stRadio div[role="radiogroup"] label:has(input:checked) {{
            background: {c["accent_soft"]} !important;
            border-color: {c["accent"]} !important;
            color: {c["text_strong"]} !important;
            font-weight: 700 !important;
        }}

        /* ── Sidebar brand ── */
        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding: 0.85rem;
            border: 1px solid {c["border"]};
            border-radius: 16px;
            background: {c["sidebar_brand_bg"]};
            box-shadow: {c["shadow"]};
        }}

        .sb-logo {{
            width: 46px;
            height: 46px;
            border-radius: 14px;
            background: linear-gradient(135deg, {c["accent"]}, {c["accent_2"]});
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
            box-shadow: 0 6px 18px rgba(15, 118, 110, 0.22);
        }}

        .sb-title {{
            font-weight: 700;
            font-size: 0.92rem;
            color: {c["text_strong"]};
            line-height: 1.25;
        }}

        .sb-sub {{
            font-size: 0.75rem;
            color: {c["muted"]};
            margin-top: 0.2rem;
        }}

        /* ── Buttons: default = light surface + dark text; primary = teal + white text ── */
        .stButton > button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stFormSubmitButton"] > button,
        [data-testid="stFormSubmitButton"] button {{
            border-radius: 12px !important;
            border: 1px solid {c["btn_secondary_border"]} !important;
            background: {c["btn_secondary_bg"]} !important;
            background-color: {c["btn_secondary_bg"]} !important;
            color: {c["btn_secondary_text"]} !important;
            -webkit-text-fill-color: {c["btn_secondary_text"]} !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.1rem !important;
            transition: all 0.18s ease;
            box-shadow: {c["shadow"]};
        }}

        .stButton > button p,
        .stButton > button span,
        .stButton > button div,
        [data-testid="stBaseButton-secondary"] p,
        [data-testid="stBaseButton-secondary"] span,
        [data-testid="stFormSubmitButton"] button p,
        [data-testid="stFormSubmitButton"] button span {{
            color: {c["btn_secondary_text"]} !important;
            -webkit-text-fill-color: {c["btn_secondary_text"]} !important;
        }}

        .stButton > button:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stFormSubmitButton"] > button:hover {{
            transform: translateY(-1px);
            background: {c["btn_secondary_hover"]} !important;
            background-color: {c["btn_secondary_hover"]} !important;
            color: {c["btn_secondary_text"]} !important;
            box-shadow: {c["shadow_lg"]};
        }}

        .stButton > button[kind="primary"],
        [data-testid="stBaseButton-primary"],
        button[data-testid="baseButton-primary"] {{
            border: 1px solid {c["btn_border"]} !important;
            background: {c["btn_bg"]} !important;
            background-color: {c["btn_bg"]} !important;
            color: {c["btn_text"]} !important;
            -webkit-text-fill-color: {c["btn_text"]} !important;
        }}

        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div,
        [data-testid="stBaseButton-primary"] p,
        [data-testid="stBaseButton-primary"] span,
        button[data-testid="baseButton-primary"] p,
        button[data-testid="baseButton-primary"] span {{
            color: {c["btn_text"]} !important;
            -webkit-text-fill-color: {c["btn_text"]} !important;
        }}

        .stButton > button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {{
            background: {c["btn_bg_hover"]} !important;
            background-color: {c["btn_bg_hover"]} !important;
            color: {c["btn_text"]} !important;
        }}

        div[data-testid="stMetric"] {{
            background: {c["card_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 16px;
            padding: 1rem 1rem;
            box-shadow: {c["shadow"]};
        }}

        div[data-testid="stMetricLabel"] {{
            color: {c["muted"]} !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: {c["text_strong"]} !important;
            font-weight: 800 !important;
        }}

        .topbar-wrap {{
            border: 1px solid {c["border"]};
            border-radius: 18px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.2rem;
            background: {c["topbar_bg"]};
            box-shadow: {c["shadow"]};
        }}

        .topbar-inner {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .topbar-brand {{
            font-size: clamp(1.1rem, 2vw, 1.45rem);
            font-weight: 800;
            color: {c["text_strong"]};
            margin: 0;
            letter-spacing: -0.03em;
        }}

        .topbar-brand span {{
            color: {c["accent"]};
        }}

        .topbar-meta {{
            font-size: 0.82rem;
            color: {c["muted"]};
            margin: 0;
        }}

        .hero-card {{
            border: 1px solid {c["border"]};
            border-radius: 20px;
            padding: 1.5rem 1.5rem;
            margin-bottom: 1.1rem;
            background:
                radial-gradient(circle at top right, {c["gradient_glow_1"]}, transparent 36%),
                {c["card_bg_alt"]};
            box-shadow: {c["shadow_lg"]};
        }}

        .hero-kicker {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: {c["accent"]};
            font-weight: 700;
            margin-bottom: 0.55rem;
        }}

        .hero-title {{
            font-size: clamp(1.7rem, 4vw, 2.6rem);
            font-weight: 800;
            color: {c["text_strong"]};
            line-height: 1.12;
            margin-bottom: 0.5rem;
        }}

        .hero-subtitle {{
            font-size: 1rem;
            color: {c["muted"]};
            line-height: 1.65;
            max-width: 780px;
        }}

        .page-card {{
            background: {c["card_bg"]};
            border: 1px solid {c["border"]};
            border-left: 4px solid {c["accent"]};
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: {c["shadow"]};
        }}

        .section-card {{
            background: {c["card_bg"]};
            border: 1px solid {c["border"]};
            border-top: 3px solid {c["accent_2"]};
            border-radius: 18px;
            padding: 1.1rem 1.1rem 0.7rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: {c["shadow"]};
        }}

        .section-title {{
            font-size: 1.02rem;
            font-weight: 700;
            color: {c["text_strong"]};
            margin-bottom: 0.3rem;
        }}

        .section-subtitle {{
            font-size: 0.88rem;
            color: {c["muted"]};
            margin-bottom: 0.9rem;
            line-height: 1.5;
        }}

        .insight-card {{
            border: 1px solid {c["border"]};
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            background: {c["card_bg"]};
            box-shadow: {c["shadow"]};
        }}

        .insight-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {c["muted"]};
            margin-bottom: 0.35rem;
            font-weight: 700;
        }}

        .insight-value {{
            font-size: 1.35rem;
            font-weight: 800;
            color: {c["text_strong"]};
            margin-bottom: 0.2rem;
        }}

        .insight-help {{
            font-size: 0.88rem;
            color: {c["muted"]};
            line-height: 1.5;
        }}

        .badge {{
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 0.35rem;
            border: 1px solid {c["border"]};
        }}

        .badge-pos {{
            background: rgba(22, 163, 74, 0.12);
            color: {pos};
        }}

        .badge-neg {{
            background: rgba(220, 38, 38, 0.10);
            color: {neg};
        }}

        .badge-neutral {{
            background: rgba(217, 119, 6, 0.10);
            color: {warn};
        }}

        .hero-header {{
            background:
                radial-gradient(circle at top right, {c["gradient_glow_1"]}, transparent 32%),
                {c["card_bg_alt"]};
            border: 1px solid {c["border"]};
            border-radius: 18px;
            padding: 1.3rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: {c["shadow"]};
        }}

        .hero-ticker {{
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: {c["text_strong"]};
        }}

        .hero-name {{
            font-size: 0.95rem;
            color: {c["muted"]};
            margin-top: 0.2rem;
        }}

        .hero-price {{
            font-size: 1.9rem;
            font-weight: 800;
            margin-top: 0.55rem;
            color: {c["text_strong"]};
        }}

        .hero-chg-pos {{
            color: {pos};
            font-weight: 700;
        }}

        .hero-chg-neg {{
            color: {neg};
            font-weight: 700;
        }}

        .rail-card {{
            background: {c["card_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            color: {c["muted"]};
            box-shadow: {c["shadow"]};
        }}

        .rail-title {{
            font-weight: 700;
            color: {c["text_strong"]};
            margin-bottom: 0.45rem;
            font-size: 0.95rem;
        }}

        .news-item {{
            padding: 0.8rem 0;
            border-bottom: 1px solid {c["border"]};
        }}

        .news-item:last-child {{
            border-bottom: none;
        }}

        .news-headline {{
            font-size: 0.92rem;
            color: {c["text_strong"]};
            font-weight: 600;
            line-height: 1.5;
        }}

        .news-meta {{
            font-size: 0.78rem;
            color: {c["muted"]};
            margin-top: 0.2rem;
        }}

        [data-testid="stTabs"] button {{
            color: {c["muted"]} !important;
            border-radius: 10px 10px 0 0 !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            border-bottom: 2px solid transparent !important;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {c["accent"]} !important;
            background: {c["accent_soft"]} !important;
            font-weight: 700 !important;
            border-bottom: 2px solid {c["accent"]} !important;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 0.25rem;
            border-bottom: 1px solid {c["border"]};
            padding-bottom: 0.25rem;
        }}

        [data-testid="stChatMessage"] {{
            background: {c["card_bg"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 14px !important;
            box-shadow: {c["shadow"]};
            padding: 0.75rem 1rem !important;
        }}

        [data-testid="stChatInput"] {{
            border-top: 1px solid {c["border"]} !important;
            padding-top: 0.75rem !important;
        }}

        [data-testid="stChatInput"] > div {{
            background: transparent !important;
        }}

        .stAlert {{
            border-radius: 12px !important;
            border: 1px solid {c["border"]} !important;
        }}

        [data-testid="stExpander"] {{
            background: {c["card_bg"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 14px !important;
            box-shadow: {c["shadow"]};
        }}

        [data-testid="stExpander"] summary {{
            color: {c["text_strong"]} !important;
            font-weight: 600 !important;
        }}

        div[data-baseweb="slider"] > div {{
            color: {c["accent"]} !important;
        }}

        .stSlider [data-baseweb="slider"] div {{
            color: {c["accent"]} !important;
        }}

        /* Toggle */
        .stToggle label span {{
            color: {c["text"]} !important;
        }}

        /* Plotly / charts */
        [data-testid="stPlotlyChart"] {{
            background: {c["card_bg"]};
            border: 1px solid {c["border"]};
            border-radius: 16px;
            padding: 0.5rem;
            box-shadow: {c["shadow"]};
        }}

        /* Dataframe */
        [data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {c["border"]};
            box-shadow: {c["shadow"]};
        }}

        /* Main markdown body text */
        .main .stMarkdown p,
        .main .stMarkdown li {{
            color: {c["text"]};
            line-height: 1.65;
        }}

        .main .stMarkdown strong {{
            color: {c["text_strong"]};
        }}

        /* Page typography */
        .page-card h2 {{
            color: {c["text_strong"]} !important;
        }}

        .page-subtitle {{
            color: {c["muted"]} !important;
        }}

        /* Feature pills row */
        .feature-pill {{
            display: inline-block;
            padding: 0.35rem 0.75rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: {c["accent_soft"]};
            color: {c["accent"]};
            border: 1px solid {c["border"]};
        }}

        /* Nav section label */
        .nav-section-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {c["muted"]};
            font-weight: 700;
            margin: 0.75rem 0 0.35rem 0.25rem;
        }}

        hr {{
            border-color: {c["border"]} !important;
            margin: 1rem 0 !important;
        }}

        /* Audio player */
        audio {{
            width: 100%;
            border-radius: 12px;
        }}

        /* Scrollbar (webkit) */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {c["border"]};
            border-radius: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle(st):
    init_theme_state(st)
    st.sidebar.toggle("Light mode", key="ui_theme_is_light")


def render_top_bar(st):
    now = datetime.now().strftime("%Y-%m-%d · %H:%M")
    first, _, rest = APP_BRAND_FULL.partition(" ")
    brand_html = '<p class="topbar-brand"><span>{}</span> {}</p>'.format(
        escape(first), escape(rest)
    )

    st.markdown(
        """
        <div class="topbar-wrap">
          <div class="topbar-inner">
            {brand}
            <p class="topbar-meta">Live dashboard · {now}</p>
          </div>
        </div>
        """.format(brand=brand_html, now=now),
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(st):
    logo_letter = escape(APP_BRAND_FULL.strip()[0].upper())
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="sb-logo">{logo}</div>
          <div>
            <div class="sb-title">{title}</div>
            <div class="sb-sub">{subtitle}</div>
          </div>
        </div>
        """.format(
            logo=logo_letter,
            title=escape(APP_BRAND_FULL),
            subtitle=escape(APP_BRAND_TAGLINE),
        ),
        unsafe_allow_html=True,
    )

    render_theme_toggle(st)

    st.sidebar.markdown('<div class="nav-section-label">Explore</div>', unsafe_allow_html=True)
    options_display = [_nav_display(*row) for row in NAV_DEFINITION]
    internal_by_display = {_nav_display(*row): row[0] for row in NAV_DEFINITION}

    choice = st.sidebar.radio(
        "nav",
        options_display,
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="nav-section-label">Capabilities</div>
        <span class="feature-pill">Live prices</span>
        <span class="feature-pill">AI analyst</span>
        <span class="feature-pill">RAG search</span>
        <span class="feature-pill">Forecasts</span>
        """,
        unsafe_allow_html=True,
    )

    return internal_by_display[choice]


def render_dashboard_hero(st):
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">✨ AI Finance Dashboard</div>
            <div class="hero-title">Track markets, sentiment, and forecasts — all in one place.</div>
            <div class="hero-subtitle">
                Real-time prices, technical indicators, news sentiment, and AI-powered analysis
                in a clean, easy-to-read workspace built for investors and researchers.
            </div>
            <div style="margin-top:1rem;">
                <span class="feature-pill">📈 Live charts</span>
                <span class="feature-pill">🤖 Agentic AI</span>
                <span class="feature-pill">📰 Sentiment</span>
                <span class="feature-pill">🔮 Forecasts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card_start(st, title, subtitle=""):
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        """.format(
            title=escape(str(title)),
            subtitle=escape(str(subtitle)),
        ),
        unsafe_allow_html=True,
    )


def render_section_card_end(st):
    st.markdown("</div>", unsafe_allow_html=True)


def render_insight_card(st, label, value, help_text):
    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-help">{help_text}</div>
        </div>
        """.format(
            label=escape(str(label)),
            value=escape(str(value)),
            help_text=escape(str(help_text)),
        ),
        unsafe_allow_html=True,
    )


def render_sentiment_badge(sentiment):
    s = str(sentiment or "").strip().lower()
    if s == "positive":
        return '<span class="badge badge-pos">Positive</span>'
    if s == "negative":
        return '<span class="badge badge-neg">Negative</span>'
    return '<span class="badge badge-neutral">Neutral</span>'


def render_company_hero(st, ticker, info):
    name = info.get("longName") or info.get("shortName") or ticker
    price = info.get("regularMarketPrice") or info.get("currentPrice")
    prev = info.get("previousClose")
    cur = info.get("currency") or "USD"

    chg_pct = None
    if price is not None and prev not in (None, 0):
        try:
            chg_pct = (float(price) - float(prev)) / float(prev) * 100.0
        except Exception:
            chg_pct = None

    chg_cls = "hero-chg-pos" if (chg_pct is not None and chg_pct >= 0) else "hero-chg-neg"
    chg_txt = ""
    if chg_pct is not None:
        sign = "+" if chg_pct >= 0 else ""
        chg_txt = '<span class="{cls}">{sign}{pct:.2f}%</span> vs prior close'.format(
            cls=chg_cls, sign=sign, pct=chg_pct
        )

    if isinstance(price, (int, float)):
        price_txt = "{:,.2f}".format(price)
    else:
        price_txt = str(price) if price else "—"

    st.markdown(
        """
        <div class="hero-header">
          <div class="hero-ticker">{ticker}</div>
          <div class="hero-name">{name}</div>
          <div class="hero-price">{price} <span style="font-size:0.95rem;font-weight:500;">{currency}</span></div>
          <div style="margin-top:0.35rem;font-size:0.92rem;">{chg}</div>
        </div>
        """.format(
            ticker=escape(str(ticker)),
            name=escape(str(name)),
            price=escape(str(price_txt)),
            currency=escape(str(cur)),
            chg=chg_txt,
        ),
        unsafe_allow_html=True,
    )


def _fmt_metric(v):
    if v is None or v == "N/A":
        return "—"
    try:
        if isinstance(v, (int, float)):
            if abs(v) >= 1e12:
                return "{:.2f}T".format(v / 1e12)
            if abs(v) >= 1e9:
                return "{:.2f}B".format(v / 1e9)
            if abs(v) >= 1e6:
                return "{:.2f}M".format(v / 1e6)
            if abs(v) >= 1000:
                return "{:,.0f}".format(v)
            return "{:,.4g}".format(v).rstrip("0").rstrip(".")
    except Exception:
        pass
    return str(v)


def render_metric_strip(st, info):
    mcap = info.get("marketCap")
    pe = info.get("forwardPE") or info.get("trailingPE")
    divy = info.get("dividendYield")

    if isinstance(divy, float) and 0 < divy < 1:
        divy_disp = "{:.2f}%".format(divy * 100)
    else:
        divy_disp = _fmt_metric(divy) if divy not in (None, "N/A") else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Market cap", _fmt_metric(mcap))
    with c2:
        st.metric("P/E", _fmt_metric(pe))
    with c3:
        st.metric("Dividend yield", divy_disp)
    with c4:
        st.metric("Beta", _fmt_metric(info.get("beta")))


def render_right_rail_placeholder(st):
    st.markdown('<div class="rail-title">Market pulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rail-card">Add top gainers, watchlist alerts, or sector heatmaps here.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rail-card">This side rail works best for summaries, signals, and quick stats.</div>',
        unsafe_allow_html=True,
    )


def page_title(st, title, subtitle=None):
    st.markdown(
        '<div class="page-card"><h2 style="margin:0;color:inherit;">{}</h2>'.format(escape(str(title))),
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            '<p class="page-subtitle" style="margin:0.4rem 0 0 0;font-size:0.95rem;line-height:1.5;">{}</p>'.format(
                escape(str(subtitle))
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_page_hero(st, icon, kicker, title, subtitle, pills=None):
    """Reusable intro banner for feature pages (replaces external GIFs/images)."""
    pills = pills or []
    pills_html = ""
    if pills:
        pills_html = '<div style="margin-top:1rem;">' + "".join(
            '<span class="feature-pill">{}</span>'.format(escape(str(p))) for p in pills
        ) + "</div>"

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">{icon} {kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            {pills}
        </div>
        """.format(
            icon=escape(str(icon)),
            kicker=escape(str(kicker)),
            title=escape(str(title)),
            subtitle=escape(str(subtitle)),
            pills=pills_html,
        ),
        unsafe_allow_html=True,
    )


__all__ = [
    "APP_BRAND_FULL",
    "APP_BRAND_TAGLINE",
    "NAV_DEFINITION",
    "init_theme_state",
    "inject_global_css",
    "render_top_bar",
    "render_sidebar_navigation",
    "render_right_rail_placeholder",
    "render_company_hero",
    "render_metric_strip",
    "render_dashboard_hero",
    "render_page_hero",
    "render_section_card_start",
    "render_section_card_end",
    "render_insight_card",
    "render_sentiment_badge",
    "page_title",
]