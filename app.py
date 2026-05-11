import subprocess
import sys
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

    from ui_theme import (
        APP_BRAND_FULL,
        init_theme_state,
        inject_global_css,
        render_top_bar,
        render_sidebar_navigation,
        render_right_rail_placeholder,
        render_company_hero,
        render_metric_strip,
        render_dashboard_hero,
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
    from pytrends.request import TrendReq

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
        fig.update_layout(
            template="plotly_dark",
            height=height,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
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

    data_dir = './inference-data/'

     
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
        keyword = [keyword]
        pytrend = TrendReq()
        pytrend.build_payload(kw_list=keyword)
        df = pytrend.interest_over_time()
        df.drop(columns=['isPartial'], inplace=True)
        df.reset_index(inplace=True)
        df.columns = ["ds", "y"]
        return df

    def make_pred_simple(df, periods):
        """
        Prophet-free forecast: polynomial trend on time + simple component views.
        Google Trends scores are clipped to [0, 100].
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

    def _stock_forecast_fallback(df_train, period_days, n_years):
        d = df_train.dropna(subset=['ds', 'y']).copy()
        if len(d) < 5:
            st.error('Not enough price history to forecast.')
            st.stop()
        t0 = d['ds'].min()
        x = (d['ds'] - t0).dt.days.values.astype(float)
        y = d['y'].values.astype(float)
        coef = np.polyfit(x, y, 1)
        poly = np.poly1d(coef)
        last = d['ds'].max()
        future = pd.date_range(last + pd.Timedelta(days=1), periods=int(period_days), freq='D')
        all_ds = pd.concat([d['ds'].reset_index(drop=True), pd.Series(future)], ignore_index=True)
        x_all = (pd.to_datetime(all_ds) - t0).dt.days.values.astype(float)
        yhat = np.maximum(poly(x_all), 1e-6)
        forecast = pd.DataFrame({'ds': pd.to_datetime(all_ds), 'yhat': yhat})
        st.subheader('Forecast data')
        st.write(forecast.tail())
        st.write(f'Forecast plot for {n_years} years (linear trend)')
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=d['ds'], y=d['y'], name='Close (historical)'))
        figp.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(dash='dash')))
        figp.update_layout(title_text='Price and linear trend forecast')
        st.plotly_chart(figp, use_container_width=True)
        st.write('Forecast components (approx.)')
        figc, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        monthly = d.set_index('ds')['y'].resample('ME').mean()
        ax1.plot(monthly.index, monthly.values, color='#D55E00')
        ax1.set_title('Monthly average close (historical)')
        ax1.grid(True, alpha=0.3)
        resid = y - poly(x)
        ax2.hist(resid, bins=30, color='#009E73', alpha=0.8)
        ax2.set_title('Residuals vs linear trend (historical)')
        ax2.grid(True, alpha=0.3)
        figc.tight_layout()
        st.pyplot(figc)


    verified = "True"
    result = APP_BRAND_FULL

    page = render_sidebar_navigation(st)


    
    
    if page == "Dashboard":
        render_dashboard_hero(st)

        snp500 = pd.read_csv("./Datasets/SP500.csv")
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

    elif page == "Google Trends with Forecast":
        page_title(st, "Search trends", "Keyword interest from Google Trends with forecast")
        st.sidebar.write("""
        ## Choose a keyword and a prediction period 
        """)
        keyword = st.sidebar.text_input("Keyword", "Amazon")
        periods = st.sidebar.slider('Prediction time in days:', 7, 365, 90)
        

        # main section
        st.write("""
        # Welcome to Trend Predictor App
        ### This app predicts the **Google Trend** you want!
        """)
        st.image('https://media.tenor.com/xfmMlSJRvdoAAAAC/google-voice-search.gif',width=350, use_container_width=True)
        st.write("Evolution of interest:", keyword)

        df = get_data(keyword)
        forecast, fig1, fig2 = make_pred(df, periods, show_fallback_caption=True)

        st.pyplot(fig1)
            
        
        st.write("Trends Over the Years and Months")
        st.pyplot(fig2)

    elif page == "About the Project":
        page_title(st, "Overview", "Data sources, architecture, and embedded dashboard")
        col_main, col_rail = st.columns([2.2, 1])
        with col_main:
            st.title('Data Sources')
            st.write("""
            ### Our F.A.S.T application have 3 data sources for two different use cases:
            #### 1. Web Scrapping to get Live News Data
            #### 2. Twitter API to get Real time Tweets
            #### 3. Google Trends API to get Real time Trends
            """)
            st.text('')


            st.title('AWS Data Architecture')
            st.image('./Images/Architecture Final AWS_FAST.jpg', width=900, use_container_width=True)

            st.title('Dashboard')
            import streamlit.components.v1 as components
            components.iframe("https://app.powerbi.com/reportEmbed?reportId=ae040e1c-7da3-4b0b-bd58-844abe577eea&autoAuth=true&ctid=a8eec281-aaa3-4dae-ac9b-9a398b9215e7", height=400, width=800)
        with col_rail:
            render_right_rail_placeholder(st)

    
    elif page == "Meeting Summarization":
        page_title(st, "Meeting notes", "Audio preview and text summary helpers")

        symbols = ['./Audio Files/Meeting 1.mp3','./Audio Files/Meeting 2.mp3', './Audio Files/Meeting 3.mp3', './Audio Files/Meeting 4.mp3']

        track = st.selectbox('Choose a the Meeting Audio',symbols)

        st.audio(track)
        data_dir = './inference-data/'

        ratiodata = st.text_input("Please Enter a Ratio you want summary by: (TRY: 0.01)")
        if st.button("Generate a Summarized Version of the Meeting"):
            time.sleep(2.4)
            #st.success("This is the Summarized text of the Meeting Audio Files xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  xxxxxxgeeeeeeeeeeeeeee eeeeeeeeeeeeeehjjjjjjjjjjjjjjjsdbjhvsdk vjbsdkvjbsdvkb skbdv")
            
            
            if track == "./Audio Files/Meeting 2.mp3":
                user_input = "NKE"
                time.sleep(1.4)
                try:
                    with open(data_dir + user_input) as f:
                        st.success(safe_summarize(f.read(), ratio=float(ratiodata)))          
                        #print()
                        st.warning("Sentiment: Negative")
                except:
                    st.text("Please Enter a valid Decimal value like 0.01")

            else:
                user_input = "AGEN"
                time.sleep(1.4)
                try:
                    with open(data_dir + user_input) as f:
                        st.success(safe_summarize(f.read(), ratio=float(ratiodata)))          
                        #print()
                        st.success("Sentiment: Positive")
                except:
                    st.text("Please Enter a valid Decimal value like 0.01")

    elif page == "Twitter Trends":
        page_title(st, "Social trends", "Interest-over-time style chart for your keyword")
        st.write("""
        # Welcome to Twitter Sentiment App
        ### This app predicts the **Twitter Sentiments** you want!
        """)
        st.image('https://assets.teenvogue.com/photos/56b4f21327a088e24b967bb6/3:2/w_531,h_354,c_limit/twitter-gifs.gif',width=350, use_container_width=True)
        ################# Twitter API Connection #######################
        
        st.sidebar.write("""
        ## Choose a keyword and a prediction period 
        """)
        keyword = st.sidebar.text_input("Keyword", "Amazon")
        periods = st.sidebar.slider('Prediction time in days:', 7, 365, 90)

        st.subheader(f"Interest over time: **{keyword}**")

        df = get_data(keyword)
        forecast, fig1, fig2 = make_pred(df, periods, show_fallback_caption=False)

        st.pyplot(fig1)
        st.write("Trends over the years and months")
        st.pyplot(fig2)



        
    elif page == "Stock Future Prediction":
        page_title(st, "Stock forecast", "Historical prices and forward projection")
        snp500 = pd.read_csv("./Datasets/SP500.csv")
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols)

        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        st.image('https://media2.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy-downsized-large.gif', width=250, use_container_width=True)

        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        data_load_state = st.text('Loading data...')

        data = safe_yf_download(ticker, START, TODAY)
        if data.empty:
            st.stop()
        data.reset_index(inplace=True)
        data_load_state.text('Loading data... done!')

        st.subheader('Raw data')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
        plot_raw_data()

        df_train = data[['Date', 'Close']].copy()
        df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})
        df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce').ffill().bfill()

        if prophet_available and plot_plotly is not None:
            try:
                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                st.subheader('Forecast data')
                st.write(forecast.tail())
                st.write(f'Forecast plot for {n_years} years')
                fig1 = plot_plotly(m, forecast)
                st.plotly_chart(fig1)
                st.write('Forecast components')
                fig2 = m.plot_components(forecast)
                st.pyplot(fig2)
            except Exception:
                st.caption('Prophet failed; using linear trend forecast instead.')
                _stock_forecast_fallback(df_train, period, n_years)
        else:
            st.caption('Using linear trend forecast (Prophet is not installed or not usable on this machine).')
            _stock_forecast_fallback(df_train, period, n_years)




    
    elif page == "Company Advanced Details":
        page_title(st, "Technicals", "Moving averages and MACD")
        snp500 = pd.read_csv("./Datasets/SP500.csv")
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols)

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

        figMACD.update_yaxes(tickprefix="$")
        st.plotly_chart(figMACD, use_container_width=True)


        






    elif page == "Live News Sentiment":
        page_title(st, "News & sentiment", "Headlines and VADER scores from Finviz")
        st.image('https://www.visitashland.com/files/latestnews.jpg', width=250, use_container_width=True)

        snp500 = pd.read_csv("./Datasets/SP500.csv")
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols)

        if st.button("Click here to See Latest News about " + ticker):
            st.header('Latest News') 

            def newsfromfizviz(temp):
                finwiz_url = 'https://finviz.com/quote.ashx?t='
                news_tables = {}
                tickers = [temp]

                for ticker in tickers:
                    url = finwiz_url + ticker
                    req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})
                    response = urlopen(req)
                    html = BeautifulSoup(response, 'html.parser')
                    news_table = html.find(id='news-table')
                    news_tables[ticker] = news_table

                parsed_news = []
                for file_name, news_table in news_tables.items():
                    for x in news_table.findAll('tr'):
                        if x.a:
                            text = x.a.get_text()
                        else:
                            text = "No headline available"
                        date_scrape = x.td.text.split()
                        if len(date_scrape) == 1:
                            time = date_scrape[0]
                        else:
                            date = date_scrape[0]
                            time = date_scrape[1]
                        ticker = file_name.split('_')[0]
                        parsed_news.append([ticker, date, time, text])

                vader = SentimentIntensityAnalyzer()
                columns = ['ticker', 'date', 'time', 'headline']
                parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
                scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
                scores_df = pd.DataFrame(scores)
                parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
                parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news['date'], errors='coerce').dt.date
                parsed_and_scored_news['Sentiment'] = np.where(parsed_and_scored_news['compound'] > 0, 'Positive', np.where(parsed_and_scored_news['compound'] == 0, 'Neutral', 'Negative'))
                return parsed_and_scored_news

            df = newsfromfizviz(ticker)
            df_pie = df[['Sentiment', 'headline']].groupby('Sentiment').count()
            fig = px.pie(df_pie, values=df_pie['headline'], names=df_pie.index, color=df_pie.index, color_discrete_map={'Positive': 'green', 'Neutral': 'darkblue', 'Negative': 'red'})

            st.subheader('Dataframe with Latest News')
            st.dataframe(df)

            st.subheader('Latest News Sentiment Distribution using Pie Chart')
            st.plotly_chart(fig)

            plt.rcParams['figure.figsize'] = [11, 5]
            mean_scores = df.groupby(['ticker', 'date']).mean(numeric_only=True)
            mean_scores = mean_scores.unstack()
            mean_scores = mean_scores.xs('compound', axis="columns").transpose()
            mean_scores.plot(kind='bar')
            plt.grid()
            st.subheader('Sentiments over Time')
            st.pyplot(plt)




    elif page == "Company Basic Details":
        snp500 = pd.read_csv("./Datasets/SP500.csv")
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols)

        try:
            info = cached_company_info(ticker)
            if isinstance(info, dict) and info:
                st.session_state[f"last_info_{ticker}"] = info
        except YFRateLimitError:
            info = st.session_state.get(f"last_info_{ticker}", {})
            if info:
                st.info("Yahoo rate limited. Showing cached company details.")
            else:
                st.error("Yahoo Finance rate limit reached. Please wait a minute and try again.")
                st.stop()
        except Exception:
            info = st.session_state.get(f"last_info_{ticker}", {})
            if info:
                st.info("Unable to refresh company details. Showing cached data.")
            else:
                st.error("Unable to fetch company details right now. Please try again shortly.")
                st.stop()
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

