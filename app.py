import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# data_dir2 = '/root/Assignment4/Assignment-Trial/Assignment-Trial/fastAPIandStreamlit/awsdownload/'


#companies = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
#@st.cache
 
def install_requirements():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)   



def main():
    # Dependencies should be installed once in the environment.
    # Re-installing on every Streamlit rerun is slow and fragile.

    import streamlit as st

    from ui.theme import (
        APP_BRAND_FULL,
        init_theme_state,
        inject_global_css,
        render_top_bar,
        render_sidebar_navigation,
        render_right_rail_placeholder,
        render_company_hero,
        render_metric_strip,
        render_dashboard_hero,
        render_page_hero,
        render_section_card_start,
        render_section_card_end,
        render_insight_card,
        render_sentiment_badge,
        page_title,
    )

    st.set_page_config(
        page_title=APP_BRAND_FULL,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_theme_state(st)
    inject_global_css(st)
    render_top_bar(st)
    from os import listdir
    from os.path import isfile, join


    import time
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    import datetime as dt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import string
    from datetime import datetime
    from datetime import date

    import matplotlib.pyplot as plt

    from config.settings import get_settings
    from config.paths import architecture_image_path

    _app_settings = get_settings()
    _transcripts_dir = str(_app_settings.inference_data_path)
    if not _transcripts_dir.endswith(("/", "\\")):
        _transcripts_dir += "/"
    _sp500_csv = str(_app_settings.sp500_path)
    _arch_image = architecture_image_path()
    from services.indicators import compute_rsi, compute_bollinger_bands
    from services.technicals import get_technicals_summary

    try:
        from prophet import Prophet
        from prophet.plot import plot_plotly
        prophet_available = True
    except Exception:
        Prophet = None
        plot_plotly = None
        prophet_available = False

    # Some environments import Prophet but fail at runtime
    # due to missing/invalid Stan backend.
    if prophet_available:
        try:
            _ = Prophet()
        except Exception:
            prophet_available = False
    import json
    import re
    import textblob
    from textblob import TextBlob
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    # import openpyxl
    import time
    #import tqdm
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    #To Hide Warnings

    from urllib.request import urlopen, Request
    import bs4
    from bs4 import BeautifulSoup
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import plotly.express as px
    #from gensim.summarization import summarize
    #from transformers import pipeline as summarize

    def apply_plotly_dark(fig, height=420):
        is_light = st.session_state.get("ui_theme_is_light", True)
        template = "plotly_white" if is_light else "plotly_dark"
        grid = "rgba(30,41,59,0.08)" if is_light else "rgba(255,255,255,0.08)"
        fig.update_layout(
            template=template,
            height=height,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1E293B" if is_light else "#E8EEF5"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
        )
        fig.update_xaxes(showgrid=True, gridcolor=grid)
        fig.update_yaxes(showgrid=True, gridcolor=grid)
        return fig

    def compute_quick_stats(info, price_df):
        stats = {
            "price": "—",
            "change_pct": "—",
            "volume": "—",
            "signal": "Neutral",
        }

        try:
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev = info.get("previousClose")
            if price is not None:
                stats["price"] = f"${float(price):,.2f}"
            if price is not None and prev not in (None, 0):
                pct = ((float(price) - float(prev)) / float(prev)) * 100
                stats["change_pct"] = f"{pct:+.2f}%"

            vol = info.get("volume")
            if vol is not None:
                if vol >= 1_000_000_000:
                    stats["volume"] = f"{vol/1_000_000_000:.2f}B"
                elif vol >= 1_000_000:
                    stats["volume"] = f"{vol/1_000_000:.2f}M"
                elif vol >= 1_000:
                    stats["volume"] = f"{vol/1_000:.1f}K"
                else:
                    stats["volume"] = str(vol)

            if price_df is not None and not price_df.empty and "Close" in price_df.columns:
                recent = price_df["Close"].dropna().tail(20)
                if len(recent) >= 5:
                    sma_5 = recent.tail(5).mean()
                    sma_20 = recent.mean()
                    if sma_5 > sma_20:
                        stats["signal"] = "Bullish"
                    elif sma_5 < sma_20:
                        stats["signal"] = "Bearish"
                    else:
                        stats["signal"] = "Neutral"
        except Exception:
            pass

        return stats

    def render_news_preview(news_df):
        if news_df is None or news_df.empty:
            st.info("No recent news available.")
            return

        preview = news_df.head(5)
        for _, row in preview.iterrows():
            badge = render_sentiment_badge(str(row.get("Sentiment", "Neutral")))
            headline = str(row.get("headline", "No headline"))
            date_val = str(row.get("date", ""))
            time_val = str(row.get("time", ""))
            st.markdown(
                f'''
                <div class="news-item">
                    <div>{badge}</div>
                    <div class="news-headline">{headline}</div>
                    <div class="news-meta">{date_val} {time_val}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    def get_news_sentiment_df(temp):
        finwiz_url = 'https://finviz.com/quote.ashx?t='
        news_tables = {}
        tickers = [temp]

        for ticker in tickers:
            url = finwiz_url + ticker
            req = Request(
                url=url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
            )
            response = urlopen(req)
            html = BeautifulSoup(response, 'html.parser')
            news_table = html.find(id='news-table')
            news_tables[ticker] = news_table

        parsed_news = []
        for file_name, news_table in news_tables.items():
            if news_table is None:
                continue
            for x in news_table.findAll('tr'):
                if x.a:
                    text = x.a.get_text()
                else:
                    text = "No headline available"

                date_scrape = x.td.text.split()
                date = None
                time_val = None

                if len(date_scrape) == 1:
                    time_val = date_scrape[0]
                else:
                    date = date_scrape[0]
                    time_val = date_scrape[1]

                ticker_name = file_name.split('_')[0]
                parsed_news.append([ticker_name, date, time_val, text])

        vader = SentimentIntensityAnalyzer()
        columns = ['ticker', 'date', 'time', 'headline']
        parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

        if parsed_and_scored_news.empty:
            return parsed_and_scored_news

        scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
        scores_df = pd.DataFrame(scores)
        parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
        parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news['date'], errors='coerce').dt.date
        parsed_and_scored_news['Sentiment'] = np.where(
            parsed_and_scored_news['compound'] > 0,
            'Positive',
            np.where(parsed_and_scored_news['compound'] == 0, 'Neutral', 'Negative')
        )
        return parsed_and_scored_news

    
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    #st.set_option('deprecation.showPyplotGlobalUse', False)

    data_dir = _transcripts_dir

     
    #page = st.sidebar.radio("Choose a page", ["Homepage", "SignUp"])
    def load_data():
    #df = data.cars()
        return 0
    df = load_data()

    def safe_summarize(text, ratio):
        """
        Lightweight fallback summarizer when external summarization
        libraries are not available.
        """
        cleaned_ratio = max(0.01, min(float(ratio), 1.0))
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return "No text available to summarize."
        keep_count = max(1, int(len(sentences) * cleaned_ratio))
        return " ".join(sentences[:keep_count])

    def normalize_market_df(df):
        if df is None or df.empty:
            return pd.DataFrame()

        normalized = df.copy()

        # Flatten MultiIndex columns from yfinance (order is often Ticker × OHLCV).
        if isinstance(normalized.columns, pd.MultiIndex):
            ohlcv = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
            level0 = set(normalized.columns.get_level_values(0).unique())
            level1 = set(normalized.columns.get_level_values(1).unique())
            if ohlcv.intersection(level1) and not ohlcv.intersection(level0):
                normalized.columns = normalized.columns.get_level_values(1)
            elif ohlcv.intersection(level0) and not ohlcv.intersection(level1):
                normalized.columns = normalized.columns.get_level_values(0)
            else:
                # Prefer the level that contains typical price field names.
                if len(ohlcv.intersection(level1)) >= len(ohlcv.intersection(level0)):
                    normalized.columns = normalized.columns.get_level_values(1)
                else:
                    normalized.columns = normalized.columns.get_level_values(0)

        return normalized

    def get_price_column(df):
        if 'Adj Close' in df.columns:
            return 'Adj Close'
        if 'Close' in df.columns:
            return 'Close'
        return None

    _FINVIZ_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    _FINVIZ_TRENDS_ALIAS = {
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "meta": "META",
        "facebook": "META",
        "netflix": "NFLX",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "amd": "AMD",
        "intel": "INTC",
        "disney": "DIS",
        "walmart": "WMT",
        "nike": "NKE",
        "coca cola": "KO",
        "coca-cola": "KO",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "bank of america": "BAC",
        "goldman": "GS",
        "berkshire": "BRK-B",
        "berkshire hathaway": "BRK-B",
    }

    def _finviz_keyword_to_ticker(keyword):
        """Map a free-text keyword to a US equity symbol for Finviz quote/news pages."""
        s = str(keyword or "").strip()
        if not s:
            return "AMZN"
        low = s.lower()
        if low in _FINVIZ_TRENDS_ALIAS:
            return _FINVIZ_TRENDS_ALIAS[low]
        compact = re.sub(r"\s+", "", s).upper()
        if re.fullmatch(r"[A-Z.\-]{1,6}", compact):
            return compact.replace(".", "-")
        return compact[:6]

    def _finviz_news_row_dates(news_table):
        """Extract calendar dates from Finviz #news-table (first column; forward-fill day)."""
        parsed = []
        last_raw = None
        for tr in news_table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            parts = tds[0].get_text(" ", strip=True).split()
            if len(parts) >= 2:
                raw = parts[0]
                last_raw = raw
            elif len(parts) == 1:
                raw = last_raw
            else:
                continue
            if not raw:
                continue
            low = str(raw).lower()
            if low == "today":
                ts = pd.Timestamp.now().normalize()
            else:
                ts = pd.to_datetime(raw, errors="coerce")
            if pd.isna(ts):
                continue
            parsed.append(ts.normalize())
        return parsed

    def _finviz_news_activity_timeseries(news_table):
        """
        Daily headline counts from Finviz news timestamps, scaled to [0, 100]
        (similar visual range to legacy Google Trends charts).
        """
        dates = _finviz_news_row_dates(news_table)
        if len(dates) < 3:
            return pd.DataFrame(columns=["ds", "y"])
        s = pd.Series(pd.to_datetime(dates).normalize())
        counts = s.value_counts().sort_index()
        counts.index = pd.to_datetime(counts.index).normalize()
        end_d = counts.index.max()
        start_d = min(counts.index.min(), end_d - pd.Timedelta(days=150))
        full_idx = pd.date_range(start_d, end_d, freq="D")
        aligned = counts.reindex(full_idx.normalize(), fill_value=0.0).astype(float)
        y = aligned.values
        ymin, ymax = float(y.min()), float(y.max())
        if ymax > ymin:
            y = 100.0 * (y - ymin) / (ymax - ymin)
        else:
            y = np.full_like(y, 50.0, dtype=float)
        return pd.DataFrame({"ds": full_idx, "y": y})

    @st.cache_data(ttl=1800, show_spinner=False)
    def cached_finviz_trends_proxy_series(keyword):
        """
        Finviz has no public Google-Trends API. We approximate 'interest' using
        headline activity on the symbol's Finviz quote page (#news-table).
        """
        ticker = _finviz_keyword_to_ticker(keyword)
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        req = Request(url, headers={"User-Agent": _FINVIZ_UA})
        with urlopen(req, timeout=25) as resp:
            soup = BeautifulSoup(resp.read(), "html.parser")
        news_table = soup.find(id="news-table")
        if news_table is None:
            raise ValueError(f"No Finviz news table for symbol {ticker}.")
        out = _finviz_news_activity_timeseries(news_table)
        if out is None or out.empty or len(out) < 5:
            raise ValueError(
                f"Not enough Finviz news history for {ticker}. "
                "Enter a valid US ticker (e.g. AMZN) or a supported company name."
            )
        return out

    @st.cache_data(ttl=1800, show_spinner=False)
    def cached_google_trends_news_series(keyword):
        """Google Trends news/web search interest for keyword/ticker."""
        from services.google_trends import fetch_google_trends_news_interest
        df, query, source = fetch_google_trends_news_interest(keyword)
        return df, query, source

    def get_search_trends_data(keyword):
        """
        Google Trends interest for the stock/company keyword, with Finviz fallback.
        """
        kw = str(keyword or "").strip() or "Amazon"
        sess_key = f"_google_trends_last_ok_{kw}"
        fb_flag = f"_google_trends_used_fallback_{kw}"
        query_key = f"_google_trends_query_{kw}"
        source_key = f"_google_trends_source_{kw}"
        reason_key = f"_google_trends_fallback_reason_{kw}"

        def _store(df, query, source, used_fallback=False, reason=""):
            st.session_state[sess_key] = df.copy()
            st.session_state[query_key] = query
            st.session_state[source_key] = source
            st.session_state[fb_flag] = used_fallback
            st.session_state[reason_key] = reason
            return df

        try:
            df, query, source = cached_google_trends_news_series(kw)
            return _store(df, query, source, used_fallback=False)
        except Exception as gt_err:
            prev = st.session_state.get(sess_key)
            if isinstance(prev, pd.DataFrame) and not prev.empty:
                st.session_state[fb_flag] = True
                st.session_state[reason_key] = str(gt_err)
                return prev.copy()

            try:
                ticker = _finviz_keyword_to_ticker(kw)
                df = cached_finviz_trends_proxy_series(kw)
                return _store(
                    df,
                    f"{ticker} (Finviz headlines)",
                    "Finviz news activity (fallback)",
                    used_fallback=True,
                    reason=str(gt_err),
                )
            except Exception:
                raise gt_err

    def _finviz_snapshot_pairs(soup):
        """Parse Finviz quote snapshot label/value pairs (HTML table, not a REST API)."""
        pairs = []
        for table in soup.find_all("table", class_=lambda c: c and "snapshot-table2" in c):
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                for i in range(0, len(tds) - 1, 2):
                    label = tds[i].get_text(strip=True)
                    if not label:
                        continue
                    val_td = tds[i + 1]
                    if label.lower() == "website":
                        a = val_td.find("a", href=True)
                        val = (a.get("href") or "").strip() if a else val_td.get_text(" ", strip=True)
                    else:
                        val = val_td.get_text(" ", strip=True)
                    pairs.append((label, val))
        return pairs

    def _finviz_parse_compact_number(text):
        if text is None:
            return None
        s = str(text).strip().replace(",", "")
        if not s or s in ("-", "N/A", "—"):
            return None
        mult = 1.0
        for suf, m in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if s.endswith(suf):
                mult = m
                s = s[: -len(suf)].strip()
                break
        try:
            return float(s) * mult
        except ValueError:
            return None

    def _finviz_parse_int_volume(text):
        n = _finviz_parse_compact_number(text)
        if n is None:
            return None
        return int(round(n))

    def _finviz_parse_percent_fraction(text):
        if not text or text in ("-", "N/A"):
            return None
        s = str(text).strip().replace(",", "")
        if s.endswith("%"):
            try:
                return float(s[:-1].strip()) / 100.0
            except ValueError:
                return None
        return None

    def _finviz_eps_next_year_estimate(pairs):
        for lab, val in pairs:
            if lab == "EPS next Y" and val and "%" not in val:
                try:
                    return float(str(val).strip().replace(",", ""))
                except ValueError:
                    continue
        return None

    def _finviz_forward_dividend_parts(text):
        if not text:
            return None, None
        m = re.search(r"([\d.]+)\s*\(([\d.]+)%\)", str(text))
        if not m:
            return None, _finviz_parse_percent_fraction(text)
        try:
            rate = float(m.group(1))
            yld = float(m.group(2)) / 100.0
            return rate, yld
        except ValueError:
            return None, None

    def _finviz_build_company_info(ticker, soup):
        pairs = _finviz_snapshot_pairs(soup)
        if not pairs:
            raise ValueError("Finviz quote page contained no snapshot table data.")

        mp = {}
        for lab, val in pairs:
            mp[lab] = val

        title_name = None
        if soup.title and soup.title.string and " - " in soup.title.string:
            title_name = soup.title.string.split(" - ", 1)[1].split(" Stock")[0].strip()

        og = soup.find("meta", attrs={"property": "og:description"})
        og_desc = (og.get("content") or "").strip() if og else ""

        price = _finviz_parse_compact_number(mp.get("Price"))
        prev_close = _finviz_parse_compact_number(mp.get("Prev Close"))
        vol = _finviz_parse_int_volume(mp.get("Volume"))
        avg_vol = _finviz_parse_compact_number(mp.get("Avg Volume"))
        mcap = _finviz_parse_compact_number(mp.get("Market Cap"))
        ev = _finviz_parse_compact_number(mp.get("Enterprise Value"))
        income = _finviz_parse_compact_number(mp.get("Income"))
        beta = _finviz_parse_compact_number(mp.get("Beta"))
        trailing_pe = _finviz_parse_compact_number(mp.get("P/E"))
        forward_pe = _finviz_parse_compact_number(mp.get("Forward P/E"))
        peg = _finviz_parse_compact_number(mp.get("PEG"))
        pb = _finviz_parse_compact_number(mp.get("P/B"))
        book_sh = _finviz_parse_compact_number(mp.get("Book/sh"))
        eps_fwd = _finviz_eps_next_year_estimate(pairs)
        div_rate, div_yield = _finviz_forward_dividend_parts(mp.get("Forward Dividend & Yield", ""))
        if div_yield is None:
            div_yield = _finviz_parse_percent_fraction(mp.get("Dividend %"))

        profit_margin = _finviz_parse_percent_fraction(mp.get("Profit Margin"))
        payout = _finviz_parse_percent_fraction(mp.get("Payout"))
        ev_sales = _finviz_parse_compact_number(mp.get("EV/Sales"))
        ev_ebitda = _finviz_parse_compact_number(mp.get("EV/EBITDA"))
        sh_out = _finviz_parse_compact_number(mp.get("Shs Outstand"))
        sh_float = _finviz_parse_compact_number(mp.get("Shs Float"))
        short_ratio = _finviz_parse_compact_number(mp.get("Short Ratio"))
        short_interest = _finviz_parse_compact_number(mp.get("Short Interest"))

        summary = og_desc or (
            f"{title_name or ticker} — snapshot fundamentals from Finviz "
            f"(sector: {mp.get('Sector', 'N/A')}, industry: {mp.get('Industry', 'N/A')})."
        )

        info = {
            "longName": title_name or mp.get("Company", ticker),
            "shortName": title_name or ticker,
            "symbol": ticker,
            "regularMarketPrice": price,
            "currentPrice": price,
            "previousClose": prev_close,
            "currency": "USD",
            "volume": vol,
            "averageVolume": int(avg_vol) if avg_vol is not None else None,
            "marketCap": int(mcap) if mcap is not None else None,
            "enterpriseValue": int(ev) if ev is not None else None,
            "trailingPE": trailing_pe,
            "forwardPE": forward_pe,
            "pegRatio": peg,
            "priceToBook": pb,
            "forwardEps": eps_fwd,
            "beta": beta,
            "bookValue": book_sh,
            "dividendRate": div_rate,
            "dividendYield": div_yield,
            "fiveYearAvgDividendYield": None,
            "payoutRatio": payout,
            "profitMargins": profit_margin,
            "enterpriseToRevenue": ev_sales,
            "enterpriseToEbitda": ev_ebitda,
            "netIncomeToCommon": int(income) if income is not None else None,
            "sector": mp.get("Sector", "N/A"),
            "industry": mp.get("Industry", "N/A"),
            "country": mp.get("Country", "N/A"),
            "exchange": mp.get("Exchange", "N/A"),
            "quoteType": "EQUITY",
            "market": mp.get("Exchange", "N/A"),
            "website": mp.get("Website", "N/A"),
            "phone": "N/A",
            "address1": "N/A",
            "city": "N/A",
            "zip": "N/A",
            "longBusinessSummary": summary,
            "sharesOutstanding": int(sh_out) if sh_out is not None else None,
            "floatShares": int(sh_float) if sh_float is not None else None,
            "sharesShort": int(short_interest) if short_interest is not None else None,
            "shortRatio": short_ratio,
            "bidSize": "N/A",
            "askSize": "N/A",
            "fullTimeEmployees": _finviz_parse_int_volume(mp.get("Employees")),
        }
        return info

    @st.cache_data(ttl=900, show_spinner=False)
    def cached_finviz_company_info(sym):
        sym = str(sym).strip().upper()
        if not sym:
            return {}
        url = f"https://finviz.com/quote.ashx?t={sym}"
        req = Request(url, headers={"User-Agent": _FINVIZ_UA})
        with urlopen(req, timeout=25) as resp:
            html = resp.read()
        soup = BeautifulSoup(html, "html.parser")
        return _finviz_build_company_info(sym, soup)

    @st.cache_data(ttl=300, show_spinner=False)
    def cached_company_info(sym):
        return yf.Ticker(sym).info

    @st.cache_data(ttl=300, show_spinner=False)
    def cached_history_data(sym, start_s=None, end_s=None):
        t = yf.Ticker(sym)
        hist = t.history(
            start=start_s,
            end=end_s,
            interval='1d',
            auto_adjust=False,
            actions=False,
        )
        return hist

    @st.cache_data(ttl=300, show_spinner=False)
    def cached_download_data(sym, start_s=None, end_s=None):
        return yf.download(
            sym,
            start=start_s,
            end=end_s,
            auto_adjust=False,
            progress=False,
        )

    def _ohlcv_from_history(sym, start=None, end=None, ticker=None):
        """Single-symbol OHLCV via Ticker.history — usually more reliable than download()."""
        start_s = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else start
        end_s = end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else end
        hist = cached_history_data(sym, start_s, end_s)
        if hist is None or hist.empty:
            return pd.DataFrame()
        out = hist.reset_index()
        # yfinance uses 'Date' or 'Datetime' for the first column depending on version
        if 'Date' not in out.columns:
            if 'Datetime' in out.columns:
                out = out.rename(columns={'Datetime': 'Date'})
            elif len(out.columns) > 0:
                first = out.columns[0]
                if str(first).lower() in ('date', 'datetime', 'index') or pd.api.types.is_datetime64_any_dtype(out[first]):
                    out = out.rename(columns={first: 'Date'})
        return normalize_market_df(out)

    def safe_yf_download(ticker, start=None, end=None, ticker_obj=None):
        sym = str(ticker).strip()
        if not sym:
            return pd.DataFrame()

        start_s = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else start
        end_s = end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else end

        try:
            data = _ohlcv_from_history(sym, start=start, end=end, ticker=ticker_obj)
            if not data.empty:
                st.session_state[f"last_prices_{sym}"] = data.copy()
                return data

            data = cached_download_data(sym, start_s, end_s)
            data = normalize_market_df(data)
            if data.empty:
                cached = st.session_state.get(f"last_prices_{sym}")
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    st.info("Using cached market data due to temporary Yahoo response issues.")
                    return cached.copy()
                st.warning("No market data returned. Please try a different ticker or try again later.")
                return pd.DataFrame()
            if 'Date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
                if len(data.columns) > 0 and data.columns[0] != 'Date':
                    data = data.rename(columns={data.columns[0]: 'Date'})
            elif 'Date' not in data.columns and len(data.columns) > 0:
                data = data.reset_index()
                if data.columns[0] != 'Date':
                    data = data.rename(columns={data.columns[0]: 'Date'})
            st.session_state[f"last_prices_{sym}"] = data.copy()
            return data
        except YFRateLimitError:
            cached = st.session_state.get(f"last_prices_{sym}")
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                st.info("Yahoo rate limited. Showing cached market data.")
                return cached.copy()
            st.error("Yahoo Finance rate limit reached. Please wait a minute and try again.")
            return pd.DataFrame()
        except Exception:
            cached = st.session_state.get(f"last_prices_{sym}")
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                st.info("Using cached market data due to temporary fetch issues.")
                return cached.copy()
            st.error("Failed to fetch market data right now. Please try again shortly.")
            return pd.DataFrame()

    def get_data(keyword):
        """
        Finviz-backed proxy for the old Google Trends series: daily news headline density
        on the Finviz quote page for the resolved ticker, scaled to [0, 100].
        """
        kw = str(keyword or "").strip() or "Amazon"
        sess_key = f"_finviz_trends_last_ok_{kw}"
        fb_flag = f"_finviz_trends_used_fallback_{kw}"
        try:
            df = cached_finviz_trends_proxy_series(kw)
            st.session_state[sess_key] = df.copy()
            st.session_state[fb_flag] = False
            return df
        except Exception:
            prev = st.session_state.get(sess_key)
            if isinstance(prev, pd.DataFrame) and not prev.empty:
                st.session_state[fb_flag] = True
                return prev.copy()
            raise

    def make_pred_simple(df, periods):
        """
        Prophet-free forecast: polynomial trend on time + simple component views.
        Series values are clipped to [0, 100] where applicable.
        """
        d = df.copy()
        d['ds'] = pd.to_datetime(d['ds'], errors='coerce')
        d['y'] = pd.to_numeric(d['y'], errors='coerce').ffill().bfill()
        d = d.dropna(subset=['ds'])
        if len(d) < 3:
            st.error('Not enough trend history to build a forecast.')
            st.stop()

        last_date = d['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=int(periods),
            freq='D',
        )
        hist_ds = d['ds'].reset_index(drop=True)
        all_ds = pd.concat([hist_ds, pd.Series(future_dates)], ignore_index=True)

        t0 = d['ds'].min()
        x = (d['ds'] - t0).dt.days.values.astype(float)
        y = d['y'].values.astype(float)
        deg = min(2, max(1, len(x) - 2))
        coef = np.polyfit(x, y, deg=deg)
        poly = np.poly1d(coef)
        x_all = (pd.to_datetime(all_ds) - t0).dt.days.values.astype(float)
        yhat = np.clip(poly(x_all), 0.0, 100.0)

        forecast = pd.DataFrame({'ds': pd.to_datetime(all_ds), 'yhat': yhat})

        fig1, ax = plt.subplots(figsize=(10, 6))
        ax.plot(d['ds'], d['y'], 'k.', label='Actual', markersize=4)
        ax.plot(forecast['ds'], forecast['yhat'], color='#0072B2', linewidth=1.5, label='Forecast (trend model)')
        ax.set_xlabel('date')
        ax.set_ylabel('trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig1.tight_layout()

        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        monthly = d.set_index('ds')['y'].resample('ME').mean()
        ax1.plot(monthly.index, monthly.values, color='#D55E00')
        ax1.set_title('Monthly average (historical)')
        ax1.grid(True, alpha=0.3)
        dow_means = d.groupby(d['ds'].dt.dayofweek, sort=True)['y'].mean()
        ax2.bar(dow_means.index, dow_means.values, color='#009E73')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax2.set_title('Average by weekday')
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        return forecast, fig1, fig2

    def make_pred(df, periods, show_fallback_caption=True):
        if prophet_available:
            try:
                prophet_basic = Prophet()
                prophet_basic.fit(df)
                future = prophet_basic.make_future_dataframe(periods=periods)
                forecast = prophet_basic.predict(future)
                fig1 = prophet_basic.plot(forecast, xlabel='date', ylabel='trend', figsize=(10, 6))
                fig2 = prophet_basic.plot_components(forecast)
                forecast = forecast[['ds', 'yhat']]
                return forecast, fig1, fig2
            except Exception:
                pass
        if show_fallback_caption:
            st.caption('Using built-in trend forecast (Prophet is not installed or not usable on this machine).')
        return make_pred_simple(df, periods)

    verified = "True"
    result = APP_BRAND_FULL

    page = render_sidebar_navigation(st)


    
    
    if page == "Dashboard":
        render_dashboard_hero(st)

        snp500 = pd.read_csv(_sp500_csv)
        symbols = snp500['Symbol'].sort_values().tolist()

        top1, top2 = st.columns([2.2, 1])

        with top1:
            ticker = st.text_input(
                "Search ticker",
                value=st.session_state.get("global_ticker_search", "AAPL")
            ).upper().strip()

        with top2:
            selected_ticker = st.selectbox(
                "Or select S&P 500 ticker",
                symbols,
                index=symbols.index(ticker) if ticker in symbols else symbols.index("AAPL") if "AAPL" in symbols else 0
            )
            ticker = selected_ticker
            st.session_state["global_ticker_search"] = ticker

        try:
            info = cached_company_info(ticker)
            if isinstance(info, dict) and info:
                st.session_state[f"last_info_{ticker}"] = info
        except Exception:
            info = st.session_state.get(f"last_info_{ticker}", {})

        start = dt.datetime.today() - dt.timedelta(365 * 2)
        end = dt.datetime.today()

        price_df = safe_yf_download(ticker, start, end)
        if price_df.empty:
            st.error("Could not load market data.")
            st.stop()

        price_df = price_df.reset_index()
        price_col = get_price_column(price_df)
        if price_col is None:
            st.error("Price column not found.")
            st.stop()

        stats = compute_quick_stats(info if isinstance(info, dict) else {}, price_df)

        render_company_hero(st, ticker, info if isinstance(info, dict) else {})
        render_metric_strip(st, info if isinstance(info, dict) else {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_insight_card(st, "Live price", stats["price"], "Latest available market price for the selected stock.")
        with c2:
            render_insight_card(st, "Daily move", stats["change_pct"], "Comparison against the previous close.")
        with c3:
            render_insight_card(st, "Volume", stats["volume"], "Reported trading volume from the latest market session.")
        with c4:
            render_insight_card(st, "Trend signal", stats["signal"], "Quick signal using recent short-term vs medium-term price behavior.")

        left, right = st.columns([2.15, 1])

        with left:
            render_section_card_start(st, "Price performance", "Two-year price history with volume context")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=price_df["Date"],
                    y=price_df[price_col],
                    mode="lines",
                    name="Price",
                    line=dict(width=3)
                )
            )

            if "Volume" in price_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=price_df["Date"],
                        y=price_df["Volume"],
                        name="Volume",
                        opacity=0.22,
                        yaxis="y2"
                    )
                )

            fig.update_layout(
                yaxis=dict(title="Price"),
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                xaxis_rangeslider_visible=False,
            )
            fig = apply_plotly_dark(fig, height=460)
            st.plotly_chart(fig, use_container_width=True)
            render_section_card_end(st)

        with right:
            render_section_card_start(st, "Company snapshot", "Quick profile and positioning")
            st.markdown(f"**Name:** {info.get('longName', ticker)}")
            st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
            st.markdown(f"**Market cap:** {info.get('marketCap', 'N/A')}")
            st.markdown(f"**Website:** {info.get('website', 'N/A')}")
            st.markdown(f"**52W High:** {info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.markdown(f"**52W Low:** {info.get('fiftyTwoWeekLow', 'N/A')}")
            render_section_card_end(st)

        row2_left, row2_mid, row2_right = st.columns([1.2, 1, 1])

        with row2_left:
            render_section_card_start(st, "Technical pulse", "Short moving average view")
            chart_df = price_df.copy()
            chart_df["SMA_20"] = chart_df[price_col].rolling(20).mean()
            chart_df["SMA_50"] = chart_df[price_col].rolling(50).mean()

            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df[price_col], name="Price", line=dict(width=2.5)))
            fig_ma.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df["SMA_20"], name="20D SMA"))
            fig_ma.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df["SMA_50"], name="50D SMA"))
            fig_ma = apply_plotly_dark(fig_ma, height=360)
            st.plotly_chart(fig_ma, use_container_width=True)
            render_section_card_end(st)

        with row2_mid:
            render_section_card_start(st, "Forecast view", "Simple forward-looking trend")
            df_train = price_df[["Date", price_col]].copy()
            df_train = df_train.rename(columns={"Date": "ds", price_col: "y"})
            df_train["ds"] = pd.to_datetime(df_train["ds"], errors="coerce")
            df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce").ffill().bfill()

            forecast_days = 90
            if prophet_available and plot_plotly is not None:
                try:
                    m = Prophet()
                    m.fit(df_train)
                    future = m.make_future_dataframe(periods=forecast_days)
                    forecast = m.predict(future)[["ds", "yhat"]]
                except Exception:
                    t0 = df_train["ds"].min()
                    x = (df_train["ds"] - t0).dt.days.values.astype(float)
                    y = df_train["y"].values.astype(float)
                    coef = np.polyfit(x, y, 1)
                    poly = np.poly1d(coef)
                    future_dates = pd.date_range(df_train["ds"].max() + pd.Timedelta(days=1), periods=forecast_days, freq="D")
                    all_ds = pd.concat([df_train["ds"].reset_index(drop=True), pd.Series(future_dates)], ignore_index=True)
                    x_all = (pd.to_datetime(all_ds) - t0).dt.days.values.astype(float)
                    forecast = pd.DataFrame({"ds": all_ds, "yhat": np.maximum(poly(x_all), 1e-6)})
            else:
                t0 = df_train["ds"].min()
                x = (df_train["ds"] - t0).dt.days.values.astype(float)
                y = df_train["y"].values.astype(float)
                coef = np.polyfit(x, y, 1)
                poly = np.poly1d(coef)
                future_dates = pd.date_range(df_train["ds"].max() + pd.Timedelta(days=1), periods=forecast_days, freq="D")
                all_ds = pd.concat([df_train["ds"].reset_index(drop=True), pd.Series(future_dates)], ignore_index=True)
                x_all = (pd.to_datetime(all_ds) - t0).dt.days.values.astype(float)
                forecast = pd.DataFrame({"ds": all_ds, "yhat": np.maximum(poly(x_all), 1e-6)})

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=df_train["ds"], y=df_train["y"], name="Historical", line=dict(width=2.5)))
            fig_fc.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(dash="dash", width=2.5)))
            fig_fc = apply_plotly_dark(fig_fc, height=360)
            st.plotly_chart(fig_fc, use_container_width=True)
            render_section_card_end(st)

        with row2_right:
            render_section_card_start(st, "News sentiment", "Recent headlines and mood")
            try:
                news_df = get_news_sentiment_df(ticker)
                if news_df is not None and not news_df.empty:
                    sentiment_counts = news_df["Sentiment"].value_counts().reset_index()
                    sentiment_counts.columns = ["Sentiment", "Count"]

                    fig_sent = px.pie(
                        sentiment_counts,
                        values="Count",
                        names="Sentiment",
                        hole=0.62
                    )
                    fig_sent = apply_plotly_dark(fig_sent, height=260)
                    st.plotly_chart(fig_sent, use_container_width=True)

                    render_news_preview(news_df)
                else:
                    st.info("No news found for this ticker.")
            except Exception:
                st.warning("Unable to load live news right now.")
            render_section_card_end(st)

        render_section_card_start(st, "Raw data", "Detailed market table")
        with st.expander("View market data"):
            st.dataframe(price_df.tail(50), use_container_width=True)
        render_section_card_end(st)

    elif page == "AI Analyst":
        from views.ai_analyst import render_ai_analyst_page
        render_ai_analyst_page(st, page_title, render_section_card_start, render_section_card_end)

    elif page == "Google Trends with Forecast":
        from services.google_trends import fetch_google_news_articles, keyword_to_trends_query

        page_title(st, "Search trends", "Google Trends news interest with forecast")
        render_page_hero(
            st,
            icon="🔎",
            kicker="Search & attention",
            title="Google Trends news interest with forecast",
            subtitle=(
                "Scrapes Google Trends **news** search interest for your stock or company, "
                "then projects the trend forward. Related headlines are pulled from Google News."
            ),
            pills=["Google Trends · news", "Stock / company query", "Prophet forecast", "Cached 30 min"],
        )
        st.sidebar.write("## Enter a company name or stock ticker")
        keyword = st.sidebar.text_input("Keyword or ticker", "Amazon")
        periods = st.sidebar.slider('Prediction time in days:', 7, 365, 90)
        st.sidebar.caption(
            "Uses **Google Trends** with the news filter (`gprop=news`) — how often people "
            "search news about this stock. Cached ~30 minutes per keyword."
        )

        trends_query = keyword_to_trends_query(keyword)
        st.info(f"Google Trends news query: **{trends_query}** (from “{keyword}”)")

        try:
            df = get_search_trends_data(keyword)
        except Exception as exc:
            st.error(
                "Could not load search trend data from Google Trends or Finviz. "
                "Try a company name (e.g. Amazon, Tesla) or ticker (e.g. AMZN, TSLA)."
            )
            with st.expander("Technical details"):
                st.code(str(exc))
            st.stop()

        kw_norm = str(keyword or "").strip() or "Amazon"
        data_source = st.session_state.get(f"_google_trends_source_{kw_norm}", "Google Trends")
        if data_source and "Finviz" in str(data_source):
            st.warning(
                f"Showing **{data_source}** — Google Trends was unavailable "
                "(rate limit or library issue). Charts use Finviz headline volume instead."
            )
            reason = st.session_state.get(f"_google_trends_fallback_reason_{kw_norm}")
            if reason:
                with st.expander("Why Google Trends failed"):
                    st.caption(reason)
        elif st.session_state.get(f"_google_trends_used_fallback_{kw_norm}"):
            st.info(
                "Could not refresh from Google Trends; showing the **last successful** series "
                "for this keyword in your current session."
            )
        else:
            st.caption(f"Data source: **{data_source}**")

        forecast, fig1, fig2 = make_pred(df, periods, show_fallback_caption=True)

        st.pyplot(fig1)

        st.write("News interest over the calendar (monthly / weekday patterns)")
        st.pyplot(fig2)

        render_section_card_start(st, "Related news articles", "Headlines from Google News for this stock")
        try:
            articles = fetch_google_news_articles(keyword, limit=12)
            if articles:
                for art in articles:
                    headline = art.get("headline", "")
                    url = art.get("url", "")
                    pub = art.get("published", "")
                    if url:
                        st.markdown(f"- [{headline}]({url})  \n  <span style='opacity:0.7;font-size:0.85em'>{pub}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"- {headline}")
            else:
                st.info("No related news articles found for this query.")
        except Exception:
            st.warning("Could not load related news articles right now.")
        render_section_card_end(st)

    elif page == "About the Project":
        page_title(st, "Overview", "Data sources, architecture, and open-source analytics dashboard")
        col_main, col_rail = st.columns([2.2, 1])
        with col_main:
            st.title("Data Sources")
            st.write("""
            ### F.A.S.T uses live and local data sources:
            - **Yahoo Finance** — prices, fundamentals, benchmarks
            - **Finviz** — news headlines and snapshot fundamentals
            - **Google Trends** — news search interest and related headlines (Search trends)
            - **X (Twitter) + Kafka** — real-time tweet stream for social trend analysis
            - **Earnings transcripts** — local corpus for RAG / AI Analyst
            - **VADER + Prophet** — sentiment scoring and forecasting
            """)

            st.title("AWS Data Architecture")
            if _arch_image:
                st.image(str(_arch_image), width=900, use_container_width=True)

            st.title("Market Dashboard")
            st.caption("Open-source dashboard built with Plotly and Streamlit — no Power BI or external login.")
            from views.overview_dashboard import render_overview_dashboard
            render_overview_dashboard(st)
        with col_rail:
            render_right_rail_placeholder(st)

    
    elif page == "Meeting Summarization":
        page_title(st, "Meeting notes", "Earnings transcript summarization with RAG")

        data_dir = _transcripts_dir
        available_tickers = sorted([
            f for f in listdir(data_dir)
            if isfile(join(data_dir, f)) and not f.startswith('.')
        ])

        ticker_pick = st.selectbox(
            "Select earnings transcript",
            available_tickers,
            index=available_tickers.index("NKE") if "NKE" in available_tickers else 0,
        )

        use_rag = st.checkbox("Use RAG + AI summary (recommended)", value=True)

        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                if use_rag:
                    try:
                        from rag.retriever import get_retriever
                        from agents.orchestrator import get_agent

                        retriever = get_retriever()
                        ctx = retriever.query_with_context(
                            f"key highlights risks outlook financial results for {ticker_pick}",
                            ticker_pick,
                        )
                        if ctx.get("context"):
                            agent = get_agent()
                            result = agent.chat(
                                f"Summarize this earnings call transcript for {ticker_pick}. "
                                f"Include sentiment (positive/negative/mixed) and key financial metrics.\n\n"
                                f"{ctx['context'][:6000]}",
                                ticker=ticker_pick,
                            )
                            st.success(result.get("response", "No summary generated."))
                            if ctx.get("sources"):
                                st.caption(f"Sources: {len(ctx['sources'])} transcript chunks via RAG")
                        else:
                            st.warning("No RAG results. Index corpus on AI Analyst → Setup, or use fallback.")
                            use_rag = False
                    except Exception as exc:
                        st.warning(f"RAG unavailable ({exc}). Falling back to extractive summary.")
                        use_rag = False

                if not use_rag:
                    ratiodata = st.session_state.get("_meeting_ratio", "0.05")
                    try:
                        with open(data_dir + ticker_pick) as f:
                            st.success(safe_summarize(f.read(), ratio=float(ratiodata)))
                    except Exception:
                        st.error("Could not read transcript file.")

        ratiodata = st.text_input(
            "Extractive ratio (fallback mode only)",
            value="0.05",
            key="_meeting_ratio",
        )

    elif page == "Twitter Trends":
        from views.x_social_trends import render_x_social_trends_page
        render_x_social_trends_page(
            st,
            page_title=page_title,
            render_page_hero=render_page_hero,
            render_section_card_start=render_section_card_start,
            render_section_card_end=render_section_card_end,
            render_sentiment_badge=render_sentiment_badge,
            make_pred=make_pred,
        )
    elif page == "Stock Future Prediction":
        from views.forecast_ui import render_stock_forecast_page
        render_stock_forecast_page(
            st,
            page_title=page_title,
            render_page_hero=render_page_hero,
            render_section_card_start=render_section_card_start,
            render_section_card_end=render_section_card_end,
            apply_plotly_theme=apply_plotly_dark,
            sp500_csv_path=_sp500_csv,
            safe_yf_download=safe_yf_download,
            prophet_available=prophet_available,
            Prophet=Prophet,
            plot_plotly=plot_plotly,
        )




    
    elif page == "Company Advanced Details":
        page_title(st, "Technicals", "SMA, EMA, MACD, RSI, and Bollinger Bands")

        snp500 = pd.read_csv(_sp500_csv)
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols,
            key="technicals_ticker",
        )

        def calcMovingAverage(data, size):
            df = data.copy()
            price_column = get_price_column(df)
            if price_column is None:
                return pd.DataFrame()
            df['sma'] = df[price_column].rolling(size).mean()
            df['ema'] = df[price_column].ewm(span=size, min_periods=size).mean()
            df.dropna(inplace=True)
            return df

        def calc_macd(data):
            df = data.copy()
            price_column = get_price_column(df)
            if price_column is None:
                return pd.DataFrame()
            df['ema12'] = df[price_column].ewm(span=12, min_periods=12).mean()
            df['ema26'] = df[price_column].ewm(span=26, min_periods=26).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
            df.dropna(inplace=True)
            return df

        st.subheader('Moving Average')

        coMA1, coMA2 = st.columns(2)

        with coMA1:
            numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    

        with coMA2:
            windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)  

        start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
        end = dt.datetime.today()
        dataMA = safe_yf_download(ticker, start, end)
        if dataMA.empty:
            st.stop()
        df_ma = calcMovingAverage(dataMA, windowSizeMA)
        if df_ma.empty:
            st.warning("Not enough data available for moving average calculation.")
            st.stop()
        df_ma = df_ma.reset_index()
        price_col_ma = get_price_column(df_ma)
        if price_col_ma is None:
            st.warning("Price column missing in downloaded data.")
            st.stop()

        figMA = go.Figure()

        figMA.add_trace(
            go.Scatter(
                x = df_ma['Date'],
                y = df_ma[price_col_ma],
                name = "Prices Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x = df_ma['Date'],
                y = df_ma['sma'],
                name = "SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x = df_ma['Date'],
                y = df_ma['ema'],
                name = "EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        figMA.update_layout(legend_title_text='Trend')
        figMA.update_yaxes(tickprefix="$")

        st.plotly_chart(figMA, use_container_width=True)  

        st.subheader('Moving Average Convergence Divergence (MACD)')
        numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2) 

        startMACD = dt.datetime.today() - dt.timedelta(numYearMACD * 365)
        endMACD = dt.datetime.today()
        dataMACD = safe_yf_download(ticker, startMACD, endMACD)
        if dataMACD.empty:
            st.stop()
        df_macd = calc_macd(dataMACD)
        if df_macd.empty:
            st.warning("Not enough data available for MACD calculation.")
            st.stop()
        df_macd = df_macd.reset_index()
        price_col_macd = get_price_column(df_macd)
        if price_col_macd is None:
            st.warning("Price column missing in downloaded data.")
            st.stop()

        figMACD = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.01)

        figMACD.add_trace(
            go.Scatter(
                x = df_macd['Date'],
                y = df_macd[price_col_macd],
                name = "Prices Over Last " + str(numYearMACD) + " Year(s)"
        
        ),
            row=1, col=1
        )

        figMACD.add_trace(
            go.Scatter(
                x = df_macd['Date'],
                y = df_macd['ema26'],
                name = "EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
            ),
            row=1, col=1
        )

        figMACD.add_trace(
            go.Scatter(
                x = df_macd['Date'],
                y = df_macd['macd'],
                name = "MACD Line"
            ),
            row=2, col=1
        )

        figMACD.add_trace(
            go.Scatter(
                x = df_macd['Date'],
                y = df_macd['signal'],
                name = "Signal Line"
            ),
            row=2, col=1
        )

        figMACD.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0
        ))

        figMACD.update_yaxes(tickprefix="$", row=1, col=1)
        figMACD = apply_plotly_dark(figMACD, height=520)
        st.plotly_chart(figMACD, use_container_width=True)

        # ── Bollinger Bands ──
        render_section_card_start(st, "Bollinger Bands", "Volatility envelope around a moving average")
        col_bb1, col_bb2, col_bb3 = st.columns(3)
        with col_bb1:
            numYearBB = st.number_input("Period (years)", min_value=1, max_value=10, value=2, key="bb_years")
        with col_bb2:
            bb_window = st.number_input("Band window (days)", min_value=10, max_value=60, value=20, key="bb_window")
        with col_bb3:
            bb_std = st.number_input("Std deviations", min_value=1.0, max_value=3.0, value=2.0, step=0.5, key="bb_std")

        start_bb = dt.datetime.today() - dt.timedelta(numYearBB * 365)
        data_bb = safe_yf_download(ticker, start_bb, dt.datetime.today())
        if not data_bb.empty:
            data_bb = data_bb.reset_index()
            price_col_bb = get_price_column(data_bb)
            if price_col_bb:
                work_bb = data_bb[["Date", price_col_bb]].copy() if "Date" in data_bb.columns else data_bb[[price_col_bb]].copy()
                work_bb[price_col_bb] = pd.to_numeric(work_bb[price_col_bb], errors="coerce")
                work_bb = work_bb.dropna()
                if not work_bb.empty:
                    close_bb = work_bb[price_col_bb]
                    bb = compute_bollinger_bands(close_bb, window=int(bb_window), num_std=float(bb_std))
                    dates_bb = work_bb["Date"] if "Date" in work_bb.columns else work_bb.index

                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(
                        x=dates_bb, y=bb["upper"], name="Upper band",
                        line=dict(color="#2563EB", width=1, dash="dot"),
                    ))
                    fig_bb.add_trace(go.Scatter(
                        x=dates_bb, y=bb["lower"], name="Lower band",
                        line=dict(color="#2563EB", width=1, dash="dot"),
                        fill="tonexty", fillcolor="rgba(99, 102, 241, 0.10)",
                    ))
                    fig_bb.add_trace(go.Scatter(
                        x=dates_bb, y=bb["middle"], name=f"SMA {int(bb_window)}",
                        line=dict(color="#F59E0B", width=1.5),
                    ))
                    fig_bb.add_trace(go.Scatter(
                        x=dates_bb, y=close_bb, name="Close",
                        line=dict(color="#0F172A" if st.session_state.get("ui_theme_is_light", True) else "#E8EEF5", width=2),
                    ))
                    fig_bb.update_layout(title=f"{ticker} — Bollinger Bands ({int(bb_window)}, {bb_std}σ)")
                    fig_bb.update_yaxes(tickprefix="$")
                    fig_bb = apply_plotly_dark(fig_bb, height=420)
                    st.plotly_chart(fig_bb, use_container_width=True)

                    tech = get_technicals_summary(data_bb, price_col_bb)
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Upper band", f"${tech.get('bb_upper', '—')}")
                    b2.metric("Middle (SMA)", f"${tech.get('bb_middle', '—')}")
                    b3.metric("Lower band", f"${tech.get('bb_lower', '—')}")
                    b4.metric("%B position", f"{tech.get('bb_percent_b', '—')}%")
        render_section_card_end(st)

        # ── RSI ──
        render_section_card_start(st, "Relative Strength Index (RSI)", "Momentum oscillator — 70 overbought, 30 oversold")
        col_rsi1, col_rsi2 = st.columns(2)
        with col_rsi1:
            numYearRSI = st.number_input("Period (years)", min_value=1, max_value=10, value=2, key="rsi_years")
        with col_rsi2:
            rsi_period = st.number_input("RSI period (days)", min_value=7, max_value=21, value=14, key="rsi_period")

        start_rsi = dt.datetime.today() - dt.timedelta(numYearRSI * 365)
        data_rsi = safe_yf_download(ticker, start_rsi, dt.datetime.today())
        if not data_rsi.empty:
            data_rsi = data_rsi.reset_index()
            price_col_rsi = get_price_column(data_rsi)
            if price_col_rsi:
                work_rsi = data_rsi[["Date", price_col_rsi]].copy() if "Date" in data_rsi.columns else data_rsi[[price_col_rsi]].copy()
                work_rsi[price_col_rsi] = pd.to_numeric(work_rsi[price_col_rsi], errors="coerce")
                work_rsi = work_rsi.dropna()
                if not work_rsi.empty:
                    close_rsi = work_rsi[price_col_rsi]
                    rsi = compute_rsi(close_rsi, period=int(rsi_period))
                    dates_rsi = work_rsi["Date"] if "Date" in work_rsi.columns else work_rsi.index

                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=dates_rsi, y=rsi, name=f"RSI ({int(rsi_period)})",
                        line=dict(color="#8B5CF6", width=2),
                        fill="tozeroy", fillcolor="rgba(139, 92, 246, 0.08)",
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF4444",
                                      annotation_text="Overbought (70)", annotation_position="right")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22C55E",
                                      annotation_text="Oversold (30)", annotation_position="right")
                    fig_rsi.add_hline(y=50, line_dash="dot", line_color="rgba(128,128,128,0.5)")
                    fig_rsi.update_layout(
                        title=f"{ticker} — RSI ({int(rsi_period)})",
                        yaxis=dict(range=[0, 100], title="RSI"),
                    )
                    fig_rsi = apply_plotly_dark(fig_rsi, height=360)
                    st.plotly_chart(fig_rsi, use_container_width=True)

                    rsi_now = float(rsi.iloc[-1]) if len(rsi) else None
                    rsi_label = "Overbought" if rsi_now and rsi_now >= 70 else "Oversold" if rsi_now and rsi_now <= 30 else "Neutral"
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Current RSI", f"{rsi_now:.1f}" if rsi_now else "—")
                    r2.metric("RSI signal", rsi_label)
                    r3.metric("Period", f"{int(rsi_period)} days")
        render_section_card_end(st)


        






    elif page == "Live News Sentiment":
        page_title(st, "News & sentiment", "Headlines and VADER scores from Finviz")
        render_page_hero(
            st,
            icon="📰",
            kicker="News & sentiment",
            title="Live headlines with AI sentiment scores",
            subtitle=(
                "Pulls the latest Finviz headlines for any S&P 500 ticker and scores each headline "
                "with VADER — positive, neutral, or negative — with charts and a full table."
            ),
            pills=["Finviz news", "VADER NLP", "Donut chart", "Timeline", "Styled table"],
        )

        snp500 = pd.read_csv(_sp500_csv)
        symbols = snp500["Symbol"].sort_values().tolist()

        ticker = st.sidebar.selectbox("Choose a S&P 500 stock", symbols, key="news_sentiment_ticker")

        load = st.button(f"Load news for {ticker}", type="primary", key="load_news_sentiment")
        if load or st.session_state.get("_news_loaded_ticker") == ticker:
            st.session_state["_news_loaded_ticker"] = ticker
            from views.news_sentiment_ui import render_news_sentiment_content
            render_news_sentiment_content(
                st, ticker, render_section_card_start, render_section_card_end
            )
        else:
            st.info("Select a ticker and click **Load news** to fetch the latest headlines and charts.")




    elif page == "Company Basic Details":
        snp500 = pd.read_csv(_sp500_csv)
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols)

        try:
            info = cached_finviz_company_info(ticker)
            if isinstance(info, dict) and info:
                st.session_state[f"last_info_{ticker}"] = info
        except Exception:
            info = st.session_state.get(f"last_info_{ticker}", {})
            if info:
                st.info("Unable to refresh from Finviz. Showing cached company details.")
            else:
                st.error(
                    "Unable to load company profile from Finviz right now "
                    "(network block, ticker not found, or page layout changed). Please try again later."
                )
                st.stop()
        st.caption("Company profile metrics are sourced from Finviz public quote pages (HTML snapshot), not Yahoo Finance `.info`.")
        page_title(st, "Company profile", "Fundamentals, narrative, and price history")
        col_cb, col_cb_rail = st.columns([2.4, 1])
        with col_cb:
            render_company_hero(st, ticker, info if isinstance(info, dict) else {})
            render_metric_strip(st, info if isinstance(info, dict) else {})
            st.markdown('** Sector **: ' + str(info.get('sector', 'N/A')))
            st.markdown('** Industry **: ' + str(info.get('industry', 'N/A')))
            st.markdown('** Phone **: ' + str(info.get('phone', 'N/A')))
            st.markdown('** Address **: ' + str(info.get('address1', 'N/A')) + ', ' + str(info.get('city', 'N/A')) + ', ' + str(info.get('zip', 'N/A')) + ', ' + str(info.get('country', 'N/A')))
            st.markdown('** Website **: ' + str(info.get('website', 'N/A')))
            st.markdown('** Business summary**')
            st.info(info.get('longBusinessSummary', 'N/A'))

            fundInfo = {
                'Enterprise Value (USD)': info.get('enterpriseValue', 'N/A'),
                'Enterprise To Revenue Ratio': info.get('enterpriseToRevenue', 'N/A'),
                'Enterprise To Ebitda Ratio': info.get('enterpriseToEbitda', 'N/A'),
                'Net Income (USD)': info.get('netIncomeToCommon', 'N/A'),
                'Profit Margin Ratio': info.get('profitMargins', 'N/A'),
                'Forward PE Ratio': info.get('forwardPE', 'N/A'),
                'PEG Ratio': info.get('pegRatio', 'N/A'),
                'Price to Book Ratio': info.get('priceToBook', 'N/A'),
                'Forward EPS (USD)': info.get('forwardEps', 'N/A'),
                'Beta ': info.get('beta', 'N/A'),
                'Book Value (USD)': info.get('bookValue', 'N/A'),
                'Dividend Rate (%)': info.get('dividendRate', 'N/A'),
                'Dividend Yield (%)': info.get('dividendYield', 'N/A'),
                'Five year Avg Dividend Yield (%)': info.get('fiveYearAvgDividendYield', 'N/A'),
                'Payout Ratio': info.get('payoutRatio', 'N/A')
            }

            fundDF = pd.DataFrame.from_dict(fundInfo, orient='index', columns=['Value'])
            st.subheader('Fundamental Info')
            st.table(fundDF)

            st.subheader('General Stock Info')
            st.markdown('** Market **: ' + str(info.get('market', 'N/A')))
            st.markdown('** Exchange **: ' + str(info.get('exchange', 'N/A')))
            st.markdown('** Quote Type **: ' + str(info.get('quoteType', 'N/A')))

            start = dt.datetime.today() - dt.timedelta(2 * 365)
            end = dt.datetime.today()
            df = safe_yf_download(ticker, start, end)
            if df.empty:
                st.stop()
            df = df.reset_index()

            price_column = get_price_column(df)
            if price_column is None:
                st.warning("Price column missing in downloaded data.")
                st.stop()

            fig = go.Figure(
                data=go.Scatter(x=df['Date'], y=df[price_column])
            )
            fig.update_layout(
                title={
                    'text': "Stock Prices Over Past Two Years",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            marketInfo = {
                "Volume": info.get('volume', 'N/A'),
                "Average Volume": info.get('averageVolume', 'N/A'),
                "Market Cap": info.get("marketCap", 'N/A'),
                "Float Shares": info.get('floatShares', 'N/A'),
                "Regular Market Price (USD)": info.get('regularMarketPrice', 'N/A'),
                'Bid Size': info.get('bidSize', 'N/A'),
                'Ask Size': info.get('askSize', 'N/A'),
                "Share Short": info.get('sharesShort', 'N/A'),
                'Short Ratio': info.get('shortRatio', 'N/A'),
                'Share Outstanding': info.get('sharesOutstanding', 'N/A')
            }

            marketDF = pd.DataFrame(data=marketInfo, index=[0])
            st.table(marketDF)
        with col_cb_rail:
            render_right_rail_placeholder(st)


    else:
        verified = "False"
        result = "Please enter valid Username, Password and Acess Token!!"

        st.title(result)

if __name__ == "__main__":
    main()

