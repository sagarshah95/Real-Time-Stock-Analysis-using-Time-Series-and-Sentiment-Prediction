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

    install_requirements()

    import streamlit as st
    from os import listdir
    from os.path import isfile, join


    import time
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import datetime as dt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import string
    from datetime import datetime
    from datetime import date

    import matplotlib.pyplot as plt

    from prophet import Prophet
    from prophet.plot import plot_plotly
    from pytrends.request import TrendReq

    import tweepy
    import json
    from tweepy import OAuthHandler
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

    
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    #st.set_option('deprecation.showPyplotGlobalUse', False)

    data_dir = './inference-data/'

     
    #page = st.sidebar.radio("Choose a page", ["Homepage", "SignUp"])
    def load_data():
    #df = data.cars()
        return 0
    df = load_data()

    def get_data(keyword):
        keyword = [keyword]
        pytrend = TrendReq()
        pytrend.build_payload(kw_list=keyword)
        df = pytrend.interest_over_time()
        df.drop(columns=['isPartial'], inplace=True)
        df.reset_index(inplace=True)
        df.columns = ["ds", "y"]
        return df

# make forecasts for a new period
    def make_pred(df, periods):
        prophet_basic = Prophet()
        prophet_basic.fit(df)
        future = prophet_basic.make_future_dataframe(periods=periods)
        forecast = prophet_basic.predict(future)
        fig1 = prophet_basic.plot(forecast, xlabel="date", ylabel="trend", figsize=(10, 6))
        fig2 = prophet_basic.plot_components(forecast)
        forecast = forecast[["ds", "yhat"]]

        return forecast, fig1, fig2


    verified = "True"
    result = "F.A.S.T. WebApp"
    st.sidebar.title(result)
    st.sidebar.write("Created By: Sagar Shah [LinkedIn](https://www.linkedin.com/in/shahsagar95/)")

    page = st.sidebar.radio("Choose a Function", ["About the Project","Live News Sentiment","Company Basic Details","Company Advanced Details","Google Trends with Forecast","Twitter Trends", "Meeting Summarization"])
    
    
    


    if page == "Google Trends with Forecast":
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
        forecast, fig1, fig2 = make_pred(df, periods)

        st.pyplot(fig1)
            
        
        st.write("Trends Over the Years and Months")
        st.pyplot(fig2)

    elif page == "About the Project":

        st.title('Data Sources')
        st.write("""
        ### Our F.A.S.T application have 3 data sources for two different use cases:
        #### 1. Web Scrapping to get Live News Data
        #### 2. Twitter API to get Real time Tweets
        #### 3. Google Trends API to get Real time Trends
        """)
        st.text('')

        link = '[Project Report](https://github.com/sagarshah95/Financial-Real-Time-Stock-Analysis-using-Sentiment-Analysis-and-Time-Series-Forecasting-AWS/blob/main/README.md)'
        st.markdown(link, unsafe_allow_html=True)

        
        st.title('AWS Data Architecture')
        st.image('./Images/Architecture Final AWS_FAST.jpg',width=900, use_container_width=True)

        st.title('Dashboard')
        import streamlit.components.v1 as components
        components.iframe("https://app.powerbi.com/reportEmbed?reportId=ae040e1c-7da3-4b0b-bd58-844abe577eea&autoAuth=true&ctid=a8eec281-aaa3-4dae-ac9b-9a398b9215e7", height=400, width = 800)

    
    elif page == "Meeting Summarization":

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
                        st.success(summarize(f.read(), ratio=float(ratiodata)))          
                        #print()
                        st.warning("Sentiment: Negative")
                except:
                    st.text("Please Enter a valid Decimal value like 0.01")

            else:
                user_input = "AGEN"
                time.sleep(1.4)
                try:
                    with open(data_dir + user_input) as f:
                        st.success(summarize(f.read(), ratio=float(ratiodata)))          
                        #print()
                        st.success("Sentiment: Positive")
                except:
                    st.text("Please Enter a valid Decimal value like 0.01")

    elif page == "Twitter Trends":
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
        

        # main section
        st.write("""
        # Welcome to Trend Predictor App
        ### This app predicts the **Google Trend** you want!
        """)
        st.image('https://media.tenor.com/xfmMlSJRvdoAAAAC/google-voice-search.gif',width=350, use_container_width=True)
        st.write("Evolution of interest:", keyword)

        df = get_data(keyword)
        forecast, fig1, fig2 = make_pred(df, periods)

        st.pyplot(fig1)
            
        
        st.write("Trends Over the Years and Months")
        st.pyplot(fig2)



        
    elif page == "Stock Future Prediction":
        snp500 = pd.read_csv("./Datasets/SP500.csv")
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols)

        START = "2015-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        st.title('Stock Forecast App')

        st.image('https://media2.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy-downsized-large.gif', width=250, use_container_width=True)

        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        data_load_state = st.text('Loading data...')

        data = yf.download(ticker, START, TODAY)
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

        # Predict forecast with Prophet.
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())
            
        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)




    
    elif page == "Company Advanced Details":
        snp500 = pd.read_csv("./Datasets/SP500.csv")
        symbols = snp500['Symbol'].sort_values().tolist()   

        ticker = st.sidebar.selectbox(
            'Choose a S&P 500 Stock',
            symbols)

        stock = yf.Ticker(ticker)

        def calcMovingAverage(data, size):
            df = data.copy()
            price_column = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df['sma'] = df[price_column].rolling(size).mean()
            df['ema'] = df[price_column].ewm(span=size, min_periods=size).mean()
            df.dropna(inplace=True)
            return df

        def calc_macd(data):
            df = data.copy()
            price_column = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df['ema12'] = df[price_column].ewm(span=12, min_periods=12).mean()
            df['ema26'] = df[price_column].ewm(span=26, min_periods=26).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
            df.dropna(inplace=True)
            return df

        def calcBollinger(data, size):
            df = data.copy()
            price_column = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            df["sma"] = df[price_column].rolling(size).mean()
            df["bolu"] = df["sma"] + 2 * df[price_column].rolling(size).std(ddof=0)
            df["bold"] = df["sma"] - 2 * df[price_column].rolling(size).std(ddof=0)
            df["width"] = df["bolu"] - df["bold"]
            df.dropna(inplace=True)
            return df

        st.title('Company Stocks Advanced Details')
        st.subheader('Moving Average')

        coMA1, coMA2 = st.columns(2)

        with coMA1:
            numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    

        with coMA2:
            windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)  

        start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
        end = dt.datetime.today()
        dataMA = yf.download(ticker, start, end)
        df_ma = calcMovingAverage(dataMA, windowSizeMA)
        df_ma = df_ma.reset_index()

        figMA = go.Figure()

        figMA.add_trace(
            go.Scatter(
                x = df_ma['Date'],
                y = df_ma['Adj Close'] if 'Adj Close' in df_ma.columns else df_ma['Close'],
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
        dataMACD = yf.download(ticker, startMACD, endMACD)
        df_macd = calc_macd(dataMACD)
        df_macd = df_macd.reset_index()

        figMACD = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.01)

        figMACD.add_trace(
            go.Scatter(
                x = df_macd['Date'],
                y = df_macd['Adj Close'] if 'Adj Close' in df_macd.columns else df_macd['Close'],
                name = "Prices Over Last " + str(numYearMACD) + " Year(s)"
        
        ))

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


        st.subheader('Bollinger Band')
        coBoll1, coBoll2 = st.columns(2)
        with coBoll1:
            numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6) 

        with coBoll2:
            windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)

        startBoll = dt.datetime.today() - dt.timedelta(numYearBoll * 365)
        endBoll = dt.datetime.today()
        dataBoll = yf.download(ticker, startBoll, endBoll)
        df_boll = calcBollinger(dataBoll, windowSizeBoll)
        df_boll = df_boll.reset_index()

        figBoll = go.Figure()
        figBoll.add_trace(
            go.Scatter(
                x = df_boll['Date'],
                y = df_boll['bolu'],
                name = "Upper Band"
            )
        )

        figBoll.add_trace(
            go.Scatter(
                x = df_boll['Date'],
                y = df_boll['sma'],
                name = "SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
            )
        )

        figBoll.add_trace(
            go.Scatter(
                x = df_boll['Date'],
                y = df_boll['bold'],
                name = "Lower Band"
            )
        )

        figBoll.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="left",
            x=0
        ))

        figBoll.update_yaxes(tickprefix="$")
        st.plotly_chart(figBoll, use_container_width=True)






    elif page == "Live News Sentiment":
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

        stock = yf.Ticker(ticker)
        info = stock.info 
        st.title('Company Basic Details')
        st.subheader(info.get('longName', 'N/A')) 
        st.markdown('** Sector **: ' + info.get('sector', 'N/A'))
        st.markdown('** Industry **: ' + info.get('industry', 'N/A'))
        st.markdown('** Phone **: ' + info.get('phone', 'N/A'))
        st.markdown('** Address **: ' + info.get('address1', 'N/A') + ', ' + info.get('city', 'N/A') + ', ' + info.get('zip', 'N/A') + ', '  +  info.get('country', 'N/A'))
        st.markdown('** Website **: ' + info.get('website', 'N/A'))
        st.markdown('** Business Summary **')
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
        st.markdown('** Market **: ' + info.get('market', 'N/A'))
        st.markdown('** Exchange **: ' + info.get('exchange', 'N/A'))
        st.markdown('** Quote Type **: ' + info.get('quoteType', 'N/A'))
        
        start = dt.datetime.today() - dt.timedelta(2 * 365)
        end = dt.datetime.today()
        df = yf.download(ticker, start, end)
        df = df.reset_index()

        # Check if 'Adj Close' column exists, otherwise use 'Close'
        if 'Adj Close' in df.columns:
            price_column = 'Adj Close'
        else:
            price_column = 'Close'

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


    else:
        verified = "False"
        result = "Please enter valid Username, Password and Acess Token!!"

        st.title(result)

if __name__ == "__main__":
    main()
