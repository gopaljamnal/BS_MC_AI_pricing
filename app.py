from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import finnhub
import numpy as np
import pandas as pd
from scipy.stats import norm
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Finnhub API Key (Replace with your API key)
finnhub_client = finnhub.Client(api_key="YOUR_FINNHUB_API_KEY")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_live_stock_price(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1d")["Close"].iloc[-1]

def fetch_live_news(ticker):
    news = finnhub_client.company_news(ticker, _from="2023-01-01", to="2023-12-31")
    return " ".join([item['headline'] for item in news[:5]]) if news else "No news found."

def get_sentiment_score(news_text):
    results = sentiment_analyzer(news_text)
    scores = [res['score'] if res['label'] == 'POSITIVE' else -res['score'] for res in results]
    return np.mean(scores)

def calculate_volatility(base_volatility, sentiment_score, scale_factor=0.05):
    return base_volatility + (sentiment_score - 0.5) * scale_factor

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def monte_carlo(S, K, T, r, sigma, simulations=10000):
    np.random.seed(42)
    prices = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.randn(simulations))
    payoffs = np.maximum(prices - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)

@app.route('/api/option_price', methods=['POST'])
def option_price():
    data = request.json
    ticker = data['ticker']
    strike_price = float(data['strike_price'])
    time_to_expiry = float(data['time_to_expiry']) / 365
    risk_free_rate = float(data['risk_free_rate']) / 100
    base_volatility = float(data['base_volatility']) / 100
    simulations = int(data['simulations'])

    stock_price = fetch_live_stock_price(ticker)
    news = fetch_live_news(ticker)
    sentiment_score = get_sentiment_score(news)
    adjusted_volatility = calculate_volatility(base_volatility, sentiment_score)

    bs_price = black_scholes(stock_price, strike_price, time_to_expiry, risk_free_rate, adjusted_volatility)
    mc_price = monte_carlo(stock_price, strike_price, time_to_expiry, risk_free_rate, adjusted_volatility, simulations)

    return jsonify({
        "stock_price": stock_price,
        "sentiment_score": sentiment_score,
        "adjusted_volatility": adjusted_volatility,
        "black_scholes_price": bs_price,
        "monte_carlo_price": mc_price
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
