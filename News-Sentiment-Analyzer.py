# ---- Cell 1: In Jupyter Notebook, ensure that plots appear inline
%matplotlib inline

# ---- Cell 2: Import Libraries and Download NLTK Data ----

import requests                # For making HTTP requests to NewsAPI
import yfinance as yf          # To fetch historical stock price data
import matplotlib.pyplot as plt  # For plotting data
import pandas as pd            # For DataFrame manipulations
import nltk                    # For natural language processing
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER for sentiment analysis
import textwrap                # For truncating/wrapping long article titles

# Download the VADER lexicon (only needs to be done once)
nltk.download('vader_lexicon')

# ---- Cell 3: Define Functions and Main Workflow ----

def fetch_news(api_key, query='finance'):
    """
    Fetches news articles from NewsAPI based on the provided query.
    
    Parameters:
      - api_key: Your NewsAPI key.
      - query: The search term (default is 'finance').
    
    Returns:
      - A list of news articles (each as a dictionary), or an empty list if an error occurs.
    """
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching news: HTTP {response.status_code}")
            return []
        news_data = response.json()
        articles = news_data.get('articles', [])
        return articles
    except Exception as e:
        print(f"Exception occurred while fetching news: {e}")
        return []


def fetch_stock_data(ticker, period='10d'):
    """
    Fetches historical stock price data for the specified ticker using yfinance.
    
    Parameters:
      - ticker: The stock ticker symbol (e.g., 'AAPL').
      - period: The time period to fetch data for (default is '10d' for 10 days).
    
    Returns:
      - A pandas DataFrame with historical stock price data, or an empty DataFrame on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period=period)
        if stock_data.empty:
            print(f"Warning: No stock data found for ticker {ticker}.")
        return stock_data
    except Exception as e:
        print(f"Exception occurred while fetching stock data: {e}")
        return pd.DataFrame()


def analyze_sentiment(articles):
    """
    Analyzes the sentiment of each news article using VADER.
    
    Also extracts the article's published date (if available) to allow aggregation by day.
    
    Parameters:
      - articles: A list of news articles.
    
    Returns:
      - A list of dictionaries containing each article's title, its compound sentiment score, and its published date.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        published_at = article.get('publishedAt', None)  # ISO format string if available
        text = f'{title} {description}' if description else title
        sentiment = sia.polarity_scores(text)
        sentiment_scores.append({
            'title': title,
            'sentiment': sentiment['compound'],
            'date': published_at
        })
    return sentiment_scores


def visualize_data(stock_data, sentiment_scores, period):
    """
    Creates three visualizations:
      1. Stock Price over Time.
      2. Sentiment of Top News Articles (with truncated article titles, sentiment scores, and an explanatory note).
      3. A dual-axis chart correlating daily average sentiment with the stock price.
    
    Parameters:
      - stock_data: DataFrame containing stock price data.
      - sentiment_scores: List of dictionaries with news article titles, sentiment scores, and published dates.
      - period: The time period examined (for display in metrics).
    """
    # Create three subplots with custom height ratios (Plot 1 and Plot 3 are larger).
    fig, axes = plt.subplots(3, 1, figsize=(32, 44), gridspec_kw={'height_ratios': [2, 3, 2]})
    
    # --- Plot 1: Stock Price over Time ---
    axes[0].plot(stock_data.index, stock_data['Close'], label='Stock Price', color='blue', marker='o')
    axes[0].set_title('Stock Price over Time', fontsize=28)
    axes[0].set_xlabel('Date', fontsize=24)
    axes[0].set_ylabel('Price', fontsize=24)
    axes[0].legend(fontsize=20)
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    
    # --- Plot 2: Sentiment of Top News Articles ---
    sorted_scores = sorted(sentiment_scores, key=lambda s: abs(s['sentiment']), reverse=True)
    top_n = min(20, len(sorted_scores))
    top_articles = sorted_scores[:top_n]
    
    sentiments = [s['sentiment'] for s in top_articles]
    titles = [s['title'] for s in top_articles]
    # Increase vertical spacing using a spacing factor.
    spacing_factor = 2.0
    y_positions = [i * spacing_factor for i in range(len(titles))]
    
    # Set y-axis limits to allow extra space.
    axes[1].set_ylim(-spacing_factor * 0.5, max(y_positions) + spacing_factor * 0.5)
    
    # Use green for positive sentiment and red for negative sentiment.
    colors = ['green' if s >= 0 else 'red' for s in sentiments]
    
    axes[1].barh(y_positions, sentiments, color=colors, align='center')
    axes[1].set_title('Sentiment of Top 20 News Articles', fontsize=28)
    axes[1].set_xlabel('Sentiment (Compound Score)', fontsize=24)
    axes[1].set_xlim([-2.0, 1.1])
    axes[1].set_yticks([])  # Remove automatic y-tick labels.
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    
    # Annotate each bar with the full article title (truncated if too long) and its sentiment score.
    for y, title, sentiment in zip(y_positions, titles, sentiments):
        truncated_title = textwrap.shorten(title, width=40, placeholder="...")
        axes[1].text(-2.1, y, truncated_title, ha='right', va='center', fontsize=18)
        if sentiment >= 0:
            x_pos = sentiment + 0.05
            ha_val = 'left'
        else:
            x_pos = sentiment - 0.05
            ha_val = 'right'
        axes[1].text(x_pos, y, f"{sentiment:.2f}", va='center', ha=ha_val, fontsize=18, color='black')
    
    # Move the explanatory note closer to the chart.
    axes[1].text(0.5, -0.20, 
                 "Note: Each bar represents one article's compound sentiment score (computed by VADER). "
                 "Green indicates positive sentiment; Red indicates negative sentiment.",
                 transform=axes[1].transAxes, fontsize=20, ha='center', va='center')
    
    # --- Plot 3: Daily Average Sentiment vs Stock Price ---
    ax3 = axes[2]
    df_sentiment = pd.DataFrame(sentiment_scores)
    if 'date' in df_sentiment.columns and not df_sentiment.empty:
        df_sentiment = df_sentiment.dropna(subset=['date'])
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date
        daily_sentiment = df_sentiment.groupby('date')['sentiment'].mean().reset_index()
    else:
        daily_sentiment = pd.DataFrame()
    
    if not daily_sentiment.empty:
        stock_data = stock_data.copy()
        stock_data['date'] = stock_data.index.date
        merged_df = pd.merge(stock_data.reset_index(), daily_sentiment, on='date', how='inner')
        if not merged_df.empty:
            bar_colors = ['green' if s >= 0 else 'red' for s in merged_df['sentiment']]
            ax3.bar(merged_df['date'], merged_df['sentiment'], color=bar_colors,
                    label='Daily Avg Sentiment', alpha=0.7)
            ax3.set_ylabel('Daily Avg Sentiment', fontsize=24)
            ax3.set_xlabel('Date', fontsize=24)
            ax3.set_title('Daily Average Sentiment vs Stock Price', fontsize=28)
            ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax3.tick_params(axis='both', which='major', labelsize=20)
            
            ax3_twin = ax3.twinx()
            ax3_twin.plot(merged_df['date'], merged_df['Close'], color='blue', marker='o', linewidth=2, label='Stock Price')
            ax3_twin.set_ylabel('Stock Price', fontsize=24)
            ax3_twin.tick_params(axis='both', which='major', labelsize=20)
            
            if len(merged_df) > 1:
                corr = merged_df['sentiment'].corr(merged_df['Close'])
                ax3.text(0.02, 0.95, f"Correlation: {corr:.2f}",
                         transform=ax3.transAxes, fontsize=20,
                         verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8))
            
            explanation = (
                "Explanation:\n"
                "  • Daily Avg Sentiment Bars: Green indicates overall positive sentiment; Red indicates negative sentiment. "
                "Values above 0 suggest positive news sentiment on that day, while values below 0 suggest negative sentiment.\n"
                "  • Blue Line: Represents the stock's closing price.\n"
                "  • A higher positive correlation indicates that days with more positive sentiment are generally associated with higher stock prices, "
                "while a negative correlation suggests the opposite trend."
            )
            ax3.text(0.02, -0.45, explanation, transform=ax3.transAxes, fontsize=20,
                     ha='left', va='center', wrap=True,
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, "Not enough overlapping data between news and stock prices.",
                     transform=ax3.transAxes, ha='center', fontsize=24)
    else:
        ax3.text(0.5, 0.5, "No published date information available in articles for causality analysis.",
                 transform=ax3.transAxes, ha='center', fontsize=24)
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.45, hspace=0.8)
    plt.show()


def main():
    
    api_key = 'Your NewsAPI key' # Replace with your actual NewsAPI key (do not hard-code it in production)
    ticker = 'AAPL'   # Change Ticker as needed for the company you want to analyze.
    period = '10d'    # Input the desired time period: eg. 10 days.
    
    # Step 1: Fetch news articles related to finance.
    articles = fetch_news(api_key, query='finance')
    if not articles:
        print("No news articles fetched. Exiting.")
        return
    
    # Step 2: Perform sentiment analysis on the fetched articles using VADER.
    sentiment_scores = analyze_sentiment(articles)
    
    # Print useful metrics.
    print("=== Metrics ===")
    print("Time Period Examined:", period)
    print("Number of Articles Analyzed:", len(articles))
    avg_sentiment = sum(s['sentiment'] for s in sentiment_scores) / len(sentiment_scores)
    print("Average Sentiment (Compound):", round(avg_sentiment, 2))
    print("Sentiment Guide: -1 = Very Negative, 0 = Neutral, 1 = Very Positive")
    print("\nRecommendation: Review the top 20 news articles in the sentiment chart to understand which news "
          "pieces may be influencing overall market sentiment. The causality chart below shows how daily "
          "sentiment trends relate to the stock's closing price, providing insights for fundamental analysis.\n\n")
    
    # Step 3: Fetch historical stock data for the specified ticker.
    stock_data = fetch_stock_data(ticker, period=period)
    if stock_data.empty:
        print("No stock data available. Exiting.")
        return
    
    # Step 4: Visualize the results.
    visualize_data(stock_data, sentiment_scores, period)
    
# --- Additional: Print a list of all articles with full titles, their sources, and publication dates ---
    print("\nArticles List:")
    for idx, article in enumerate(articles, 1):
        title = article.get('title', 'No Title Provided')
        source = article.get('source', {}).get('name', 'Unknown Source')
        published_at = article.get('publishedAt', 'No Date Provided')
        print(f"{idx}. {title} - Source: {source} - Date: {published_at}")
        
# Execute the main function when running the cell
if __name__ == "__main__":
    main()
