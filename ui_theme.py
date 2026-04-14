"""
Streamlit UI theme: dark-first fintech styling, top bar, sidebar navigation.
Logic in app.py is unchanged; this module only handles presentation helpers.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

# Product branding (shown in browser tab, top bar, and sidebar)
APP_BRAND_FULL = "Financial Analysis and Stock Trading Analysis"
APP_BRAND_TAGLINE = "Real-time market insights"

# (internal_page_key, icon, short_label) — internal keys must match app.py branches
NAV_DEFINITION = [
    ("About the Project", "🏠", "Overview"),
    ("Live News Sentiment", "📰", "News & sentiment"),
    ("Company Basic Details", "📋", "Company profile"),
    ("Company Advanced Details", "📊", "Technicals"),
    ("Google Trends with Forecast", "🔎", "Search trends"),
    ("Twitter Trends", "𝕏", "Social trends"),
    ("Meeting Summarization", "🎙️", "Meeting notes"),
]


def _nav_display(internal: str, icon: str, short: str) -> str:
    return f"{icon}  {short}"


def init_theme_state(st) -> None:
    if "ui_theme_is_light" not in st.session_state:
        st.session_state["ui_theme_is_light"] = False


def inject_global_css(st) -> None:
    theme = "light" if st.session_state.get("ui_theme_is_light", False) else "dark"
    if theme == "light":
        bg = "#F6F8FA"
        panel = "#FFFFFF"
        panel2 = "#EAEEF2"
        text = "#1F2328"
        muted = "#656D76"
        border = "rgba(31,35,40,0.12)"
        shadow = "0 8px 24px rgba(31,35,40,0.08)"
    else:
        bg = "#0E1117"
        panel = "#161B22"
        panel2 = "#1C2128"
        text = "#F0F6FC"
        muted = "#8B949E"
        border = "rgba(240,246,252,0.08)"
        shadow = "0 8px 32px rgba(0,0,0,0.45)"

    pos = "#00C853"
    neg = "#FF1744"
    accent = "#3FB950"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
            color: {text};
        }}
        .stApp {{
            background-color: {bg} !important;
        }}
        /* Streamlit default chrome — without this, a light/white band hides the top of the app */
        header[data-testid="stHeader"] {{
            background-color: {bg} !important;
            background-image: none !important;
        }}
        div[data-testid="stToolbar"] {{
            background-color: {bg} !important;
        }}
        div[data-testid="stDecoration"] {{
            background-image: none !important;
            background-color: {bg} !important;
        }}
        [data-testid="collapsedControl"] {{
            color: {text} !important;
        }}
        /* Main content blocks inherit app background (fixes white strips in column layouts) */
        .main .block-container {{
            background-color: transparent !important;
        }}
        div[data-testid="column"] {{
            background-color: transparent !important;
        }}
        section.main > div,
        [data-testid="stAppViewContainer"] > .main {{
            background-color: {bg} !important;
        }}
        /* Text inputs — default Streamlit/BaseWeb is bright white */
        .stTextInput input,
        .stTextInput > div > div > input,
        [data-baseweb="input"] {{
            background-color: {panel2} !important;
            color: {text} !important;
            border-color: {border} !important;
            border-radius: 10px !important;
        }}
        .stTextInput label {{
            color: {muted} !important;
        }}
        section[data-testid="stSidebar"] > div {{
            background-color: {panel} !important;
            border-right: 1px solid {border} !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: {text} !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {{
            color: {muted} !important;
        }}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
            color: {text} !important;
        }}
        div[data-testid="stVerticalBlock"] > div:has(> label) {{
            color: {text};
        }}
        .block-container {{
            padding-top: 1.25rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }}
        h1, h2, h3 {{
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }}
        .stButton > button {{
            border-radius: 10px !important;
            font-weight: 600 !important;
            border: 1px solid {border} !important;
            background: {panel2} !important;
            color: {text} !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: {shadow};
            border-color: {accent} !important;
        }}
        div[data-testid="stMetric"] {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 0.75rem 1rem;
            box-shadow: {shadow};
        }}
        div[data-testid="stMetric"]:hover {{
            border-color: {accent};
        }}
        .topbar-wrap {{
            background: linear-gradient(180deg, {panel} 0%, {bg} 100%);
            border: 1px solid {border};
            border-radius: 14px;
            padding: 0.65rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: {shadow};
        }}
        .topbar-inner {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        .topbar-brand {{
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: {text};
            margin: 0;
        }}
        .topbar-brand span {{
            color: {accent};
        }}
        .topbar-brand-long {{
            font-size: clamp(0.72rem, 1.65vw, 0.92rem);
            line-height: 1.3;
            max-width: min(440px, 58vw);
            flex: 1 1 auto;
        }}
        .topbar-meta {{
            font-size: 0.8rem;
            color: {muted};
            white-space: nowrap;
        }}
        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.5rem;
        }}
        .sb-logo {{
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: linear-gradient(135deg, {accent}, #238636);
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
        }}
        .sb-title {{ font-weight: 700; font-size: 1rem; color: {text}; }}
        .sb-title-long {{
            font-weight: 600;
            font-size: 0.74rem;
            line-height: 1.35;
            max-width: 188px;
        }}
        .sb-sub {{ font-size: 0.72rem; color: {muted}; margin-top: 2px; }}
        .page-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: {shadow};
        }}
        .hero-header {{
            background: linear-gradient(135deg, {panel2} 0%, {panel} 100%);
            border: 1px solid {border};
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: {shadow};
        }}
        .hero-ticker {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.04em; color: {text}; }}
        .hero-name {{ font-size: 1rem; color: {muted}; margin-top: 0.25rem; }}
        .hero-price {{ font-size: 1.75rem; font-weight: 700; margin-top: 0.5rem; color: {text}; }}
        .hero-chg-pos {{ color: {pos}; font-weight: 600; }}
        .hero-chg-neg {{ color: {neg}; font-weight: 600; }}
        .rail-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
            font-size: 0.82rem;
            color: {muted};
        }}
        .rail-title {{ font-weight: 600; color: {text}; margin-bottom: 0.35rem; font-size: 0.88rem; }}
        .stRadio > div {{ gap: 0.35rem; }}
        .stRadio label {{
            border-radius: 10px !important;
            padding: 0.4rem 0.5rem !important;
            border: 1px solid transparent;
        }}
        .stRadio label:hover {{
            background: {panel2} !important;
            border-color: {border} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle(st) -> None:
    init_theme_state(st)
    st.sidebar.toggle("Light mode", key="ui_theme_is_light")


def render_top_bar(st) -> None:
    now = datetime.now().strftime("%Y-%m-%d · %H:%M")
    first, _, rest = APP_BRAND_FULL.partition(" ")
    brand_html = (
        f'<p class="topbar-brand topbar-brand-long">'
        f'<span>{escape(first)}</span> {escape(rest)}</p>'
    )
    st.markdown(
        f"""
        <div class="topbar-wrap">
          <div class="topbar-inner">
            {brand_html}
            <p class="topbar-meta">{now}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Avoid an empty leading column (can render as a large light block in some Streamlit versions)
    c_search, c_meta = st.columns([3.2, 1])
    with c_search:
        st.text_input(
            "Ticker lookup",
            placeholder="Quick search ticker (e.g. MSFT)",
            key="global_ticker_search",
            label_visibility="collapsed",
        )
    with c_meta:
        st.caption("Notifications · —")


def render_sidebar_navigation(st) -> str:
    logo_letter = escape(APP_BRAND_FULL.strip()[0].upper())
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
          <div class="sb-logo">{logo_letter}</div>
          <div>
            <div class="sb-title sb-title-long">{escape(APP_BRAND_FULL)}</div>
            <div class="sb-sub">{escape(APP_BRAND_TAGLINE)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_theme_toggle(st)
    st.sidebar.markdown("##### Navigate")
    options_display = [_nav_display(*row) for row in NAV_DEFINITION]
    internal_by_display = {_nav_display(*row): row[0] for row in NAV_DEFINITION}
    choice = st.sidebar.radio(
        "nav",
        options_display,
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Roadmap")
    st.sidebar.caption("Portfolio · Alerts · Screener (coming soon)")
    return internal_by_display[choice]


def render_company_hero(st, ticker: str, info: dict) -> None:
    name = info.get("longName") or info.get("shortName") or ticker
    price = info.get("regularMarketPrice") or info.get("currentPrice")
    prev = info.get("previousClose")
    cur = info.get("currency") or "USD"
    chg_pct = None
    if price is not None and prev not in (None, 0):
        try:
            chg_pct = (float(price) - float(prev)) / float(prev) * 100.0
        except (TypeError, ValueError):
            chg_pct = None
    chg_cls = "hero-chg-pos" if (chg_pct is not None and chg_pct >= 0) else "hero-chg-neg"
    chg_txt = ""
    if chg_pct is not None:
        sign = "+" if chg_pct >= 0 else ""
        chg_txt = f'<span class="{chg_cls}">{sign}{chg_pct:.2f}%</span> vs prior close'
    price_txt = f"{price:,.2f}" if isinstance(price, (int, float)) else (str(price) if price else "—")

    st.markdown(
        f"""
        <div class="hero-header">
          <div class="hero-ticker">{ticker}</div>
          <div class="hero-name">{name}</div>
          <div class="hero-price">{price_txt} <span style="font-size:0.95rem;font-weight:500;">{cur}</span></div>
          <div style="margin-top:0.35rem;font-size:0.9rem;">{chg_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    b1, b2, _ = st.columns([1, 1, 2])
    with b1:
        st.button("Add to watchlist", key="watchlist_btn", disabled=True, help="Coming soon")
    with b2:
        st.button("Export snapshot", key="export_btn", disabled=True, help="Coming soon")


def _fmt_metric(v) -> str:
    if v is None or v == "N/A":
        return "—"
    try:
        if isinstance(v, (int, float)):
            if abs(v) >= 1e12:
                return f"{v / 1e12:.2f}T"
            if abs(v) >= 1e9:
                return f"{v / 1e9:.2f}B"
            if abs(v) >= 1e6:
                return f"{v / 1e6:.2f}M"
            if abs(v) >= 1000:
                return f"{v:,.0f}"
            return f"{v:,.4g}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        pass
    return str(v)


def render_metric_strip(st, info: dict) -> None:
    mcap = info.get("marketCap")
    pe = info.get("forwardPE") or info.get("trailingPE")
    divy = info.get("dividendYield")
    if isinstance(divy, float) and 0 < divy < 1:
        divy_disp = f"{divy * 100:.2f}%"
    else:
        divy_disp = _fmt_metric(divy) if divy not in (None, "N/A") else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Market cap", _fmt_metric(mcap))
    with c2:
        st.metric("P/E (fwd/trail)", _fmt_metric(pe))
    with c3:
        st.metric("Dividend yield", divy_disp)
    with c4:
        st.metric("Beta", _fmt_metric(info.get("beta")))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("52W high", _fmt_metric(info.get("fiftyTwoWeekHigh")))
    with c6:
        st.metric("52W low", _fmt_metric(info.get("fiftyTwoWeekLow")))
    with c7:
        st.metric("EPS (fwd)", _fmt_metric(info.get("forwardEps")))
    with c8:
        st.metric("Volume", _fmt_metric(info.get("volume")))


def render_right_rail_placeholder(st) -> None:
    st.markdown('<div class="rail-title">Market pulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rail-card">Top movers, heatmap, and headline feed can plug in here.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rail-card">Use your data providers or cache to populate gainers/losers.</div>',
        unsafe_allow_html=True,
    )


def page_title(st, title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="page-card"><h2 style="margin:0;">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<p style="margin:0.35rem 0 0 0;opacity:0.75;font-size:0.95rem;">{subtitle}</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
