#!/usr/bin/env python3
"""
Improved Stock Market Data Collector:
Downloads free daily market data from multiple sources,
saves it as a CSV file, calculates technical indicators,
and aggregates sentiment data.
"""

import csv
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import numpy as np
import re
from pathlib import Path
from newsapi import NewsApiClient  # pip install newsapi-python
import praw  # pip install praw
import finnhub  # pip install finnhub-python
import logging

logger = logging.getLogger(__name__)

# ------------------------ Market Data & Technicals ------------------------


def get_stock_data(ticker, period="max"):
    """Fetch maximum available historical data for a single ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None, f"No data found for ticker {ticker}."
        try:
            info = stock.info
            print(f"\nCompany: {info.get('longName', ticker)}")
            print(f"Sector: {info.get('sector', 'N/A')}")
            print(f"Industry: {info.get('industry', 'N/A')}")
            print(f"Market Cap: ${info.get('marketCap', 0):,}")
        except Exception as e:
            print(f"Could not fetch company info: {e}")
        return data, None
    except Exception as e:
        return None, f"Error fetching data for {ticker}: {e}"


def calculate_sma(data, window=20):
    """Calculates Simple Moving Average."""
    return data["Close"].rolling(window=window).mean()


def calculate_ema(data, window=20):
    """Calculates Exponential Moving Average."""
    return data["Close"].ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index."""
    delta = data["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculates Moving Average Convergence Divergence."""
    ema_short = data["Close"].ewm(span=short_window, adjust=False).mean()
    ema_long = data["Close"].ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    rolling_mean = data["Close"].rolling(window=window).mean()
    rolling_std = data["Close"].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band


def calculate_historical_volatility(data, window=30):
    """Calculates historical volatility."""
    if data is None or len(data) < window:
        return "N/A", "Insufficient data for volatility calculation."
    daily_returns = data["Close"].pct_change().dropna()
    if daily_returns.empty:
        return "N/A", "Could not calculate daily returns."
    volatility = daily_returns.rolling(window=window).std() * np.sqrt(252)
    return volatility.iloc[-1], None


def get_vix_index():
    """Fetches VIX index data using yfinance (short period for near real-time)."""
    vix_data, error = get_stock_data("^VIX", period="5d")
    if vix_data is not None and not vix_data.empty:
        latest_vix = vix_data["Close"].iloc[-1]
        return latest_vix, None
    return "N/A", error


# ------------------------ Sentiment Analysis ------------------------


def analyze_sentiment(text_list):
    """Analyzes sentiment of a list of texts using TextBlob."""
    if not text_list:
        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "overall_sentiment": "Neutral",
        }
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_polarity = 0
    for text in text_list:
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            total_polarity += polarity
            if polarity > 0.1:
                positive_count += 1
            elif polarity < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        except Exception as e:
            print(f"Error analyzing text: {e}")
            continue
    total_texts = len(text_list)
    overall_polarity = total_polarity / total_texts if total_texts > 0 else 0
    overall_sentiment = (
        "Positive"
        if overall_polarity > 0.1
        else "Negative"
        if overall_polarity < -0.1
        else "Neutral"
    )
    return {
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "overall_sentiment": overall_sentiment,
        "average_polarity": overall_polarity,
    }


def fetch_news_api(ticker: str) -> List[Dict]:
    """Fetch news using NewsAPI."""
    try:
        newsapi = NewsApiClient(api_key="YOUR_NEWS_API_KEY")  # Replace with your key
        news = newsapi.get_everything(
            q=f"{ticker} stock", language="en", sort_by="relevancy", page_size=10
        )
        return news["articles"]
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return []


def fetch_reddit_api(ticker: str) -> List[Dict]:
    """Fetch Reddit posts using PRAW."""
    try:
        reddit = praw.Reddit(
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
            user_agent="YOUR_USER_AGENT",
        )
        subreddits = ["wallstreetbets", "stocks", "investing"]
        posts = []
        for subreddit in subreddits:
            for post in reddit.subreddit(subreddit).search(ticker, limit=5):
                posts.append(
                    {"title": post.title, "body": post.selftext, "score": post.score}
                )
        return posts
    except Exception as e:
        logger.error(f"Reddit API error: {e}")
        return []


def fetch_finnhub_data(ticker: str) -> List[Dict]:
    """Fetch social sentiment from Finnhub."""
    try:
        finnhub_client = finnhub.Client(api_key="YOUR_FINNHUB_API_KEY")
        sentiment = finnhub_client.social_sentiment(ticker)
        return sentiment
    except Exception as e:
        logger.error(f"Finnhub API error: {e}")
        return []


def clean_tweet_text(text):
    """Cleans tweet text by removing URLs, mentions, and hashtags."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    return text.strip()


def fetch_webpage_content(url):
    """Fetches webpage content."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Webpage fetch error: {e}")
        return None


def get_tweet_sentiment(ticker_symbol):
    """Fetches recent tweets about a stock using DuckDuckGo and analyzes sentiment."""
    search_query = f"{ticker_symbol} stock"
    search_url = "https://duckduckgo.com/html/?" + urlencode(
        {"q": search_query, "t": "social"}
    )
    html_content = fetch_webpage_content(search_url)
    if not html_content:
        return {"error": "Could not fetch tweet search results."}, None
    soup = BeautifulSoup(html_content, "html.parser")
    tweets = []
    results = soup.find_all("div", class_="cw-df__text")
    for result in results:
        tweet_text = result.get_text(strip=True)
        if tweet_text:
            cleaned_text = clean_tweet_text(tweet_text)
            tweets.append(cleaned_text)
    sentiment_analysis = analyze_sentiment(tweets)
    return sentiment_analysis, tweets if tweets else None


def get_market_sentiment(ticker_symbol):
    """Aggregates market sentiment from news, tweets, and Reddit."""
    news_articles = fetch_news_api(ticker_symbol)
    reddit_posts = fetch_reddit_api(ticker_symbol)
    finnhub_data = fetch_finnhub_data(ticker_symbol)

    news_texts = [
        article["title"] + " " + article.get("description", "")
        for article in news_articles
    ]
    reddit_texts = [post["title"] + " " + post["body"] for post in reddit_posts]

    sentiment_news = analyze_sentiment(news_texts)
    sentiment_reddit = analyze_sentiment(reddit_texts)
    sentiment_tweets, tweets = get_tweet_sentiment(ticker_symbol)

    market_sentiment = {
        "news": sentiment_news,
        "reddit": sentiment_reddit,
        "tweets": sentiment_tweets,
        "social": finnhub_data,
    }
    sources = {
        "news_articles": news_articles,
        "reddit_posts": reddit_posts,
        "tweets": tweets,
        "social_metrics": finnhub_data,
    }
    return market_sentiment, sources


# ------------------------ Market Data Download Functions ------------------------


def download_free_market_data(ticker: str) -> pd.DataFrame:
    """
    Downloads free daily market data for the given ticker from Stooq,
    using additional parameters to force the CSV to include a header and proper columns.
    """
    base_url = "https://stooq.com/q/d/l/"
    params = {
        "s": ticker.lower(),
        "i": "d",
        "f": "sd2ohlcv",  # Format: Symbol, Date, Open, High, Low, Close, Volume
        "h": "1",  # Include header row
    }
    url = f"{base_url}?{urlencode(params)}"
    print(f"Downloading market data for {ticker} from Stooq: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        csv_text = response.text
        with open("data.csv", "w") as f:
            f.write(csv_text)
        # Detect delimiter using csv.Sniffer
        sample = csv_text[:1024]
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","
        df = pd.read_csv("data.csv", delimiter=delimiter)
        # Check for a header column named "Date" (case-insensitive)
        date_col = None
        for col in df.columns:
            if col.lower() == "date":
                date_col = col
                break
        if date_col is None:
            raise ValueError("CSV does not contain 'Date' column.")
        if date_col != "Date":
            df.rename(columns={date_col: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Ticker"] = ticker
        df["Interval"] = "1d"
        return df
    except Exception as e:
        print(f"Error downloading market data from Stooq for {ticker}: {e}")
        return pd.DataFrame()


def download_yahoo_market_data(ticker: str) -> pd.DataFrame:
    """
    Downloads historical market data for the given ticker using yfinance's API
    (this avoids the direct CSV download and potential 429 errors).
    """
    print(f"Downloading market data for {ticker} from Yahoo Finance via yfinance...")
    try:
        data, err = get_stock_data(ticker, period="max")
        if data is None or data.empty:
            raise ValueError(err or "No data from Yahoo Finance")
        df = data.reset_index()  # Ensure Date becomes a column
        df["Ticker"] = ticker
        df["Interval"] = "1d"
        return df
    except Exception as e:
        print(f"Error downloading market data from Yahoo Finance for {ticker}: {e}")
        return pd.DataFrame()


def download_market_data_from_sources(ticker: str) -> pd.DataFrame:
    """
    Attempts to download market data from multiple sources.
    Tries Stooq first (with enhanced parameters) and then Yahoo Finance via yfinance.
    """
    for source in [download_free_market_data, download_yahoo_market_data]:
        df = source(ticker)
        if not df.empty:
            return df
    return pd.DataFrame()


# ------------------------ Data Collection & Saving ------------------------


def collect_comprehensive_data(ticker: str, save_dir: str = "stock_data") -> Path:
    """
    Collects comprehensive stock data by attempting multiple sources,
    calculating technical indicators, and saving the result as CSV.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    print(f"\nCollecting comprehensive data for {ticker}...")

    market_data = download_market_data_from_sources(ticker)
    if market_data.empty:
        print(f"No market data downloaded for {ticker}.")
        return None

    # Calculate technical indicators on the downloaded data
    market_data["SMA_20"] = calculate_sma(market_data, 20)
    market_data["EMA_20"] = calculate_ema(market_data, 20)
    market_data["RSI"] = calculate_rsi(market_data)
    macd_line, signal_line, macd_hist = calculate_macd(market_data)
    market_data["MACD_line"] = macd_line
    market_data["MACD_signal"] = signal_line
    market_data["MACD_hist"] = macd_hist
    upper_band, lower_band = calculate_bollinger_bands(market_data)
    market_data["BB_upper"] = upper_band
    market_data["BB_lower"] = lower_band

    # Hard-coded company info as fallback (replace with dynamic lookup if desired)
    companies = {
        "ORCL": {
            "longName": "Oracle Corporation",
            "sector": "Technology",
            "industry": "Software - Infrastructure",
            "marketCap": 487705903104,
        }
    }
    info = companies.get(
        ticker, {"longName": ticker, "sector": "N/A", "industry": "N/A", "marketCap": 0}
    )
    print(f"\nCompany: {info.get('longName', ticker)}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Market Cap: ${info.get('marketCap', 0):,}")

    print("\nAnalyzing market sentiment...")
    sentiment, sources = get_market_sentiment(ticker)

    # Save the processed data as CSV
    data_csv_path = save_path / f"{ticker}_data.csv"
    market_data.to_csv(data_csv_path, index=False)
    print(f"\nData saved to: {data_csv_path}")

    return data_csv_path


# ------------------------ Main Execution ------------------------

if __name__ == "__main__":
    print("Stock Market Data Collector")
    print("Enter 'quit' to exit\n")

    while True:
        ticker = input("Enter stock ticker symbol: ").upper().strip()
        if ticker.lower() == "quit" or ticker == "":
            break

        data_file = collect_comprehensive_data(ticker)
        if data_file:
            print(f"\nData collection complete for {ticker}.")
        else:
            print(f"\nFailed to collect data for {ticker}. Try another ticker?")

        print("\nWould you like to analyze another ticker? (Enter 'quit' to exit)")

    print("\nData collection completed.")
