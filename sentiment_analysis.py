import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime, timedelta

# Convert relative times to datetime objects
def convert_relative_time(relative_time):
    if "Min Ago" in relative_time:
        minutes = int(relative_time.split(" ")[0])
        return datetime.now() - timedelta(minutes=minutes)
    elif "Hours Ago" in relative_time:
        hours = int(relative_time.split(" ")[0])
        return datetime.now() - timedelta(hours=hours)
    else:
        return datetime.now()

# Updated Yahoo Finance Scraper (More Reliable Headline Extraction)
def get_yahoo_finance_articles():
    url = "https://finance.yahoo.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    # New approach: Grab all 'h3' tags directly for headlines
    for item in soup.find_all("h3"):
        headline = item.get_text(strip=True)
        if headline:
            headlines.append(("Yahoo Finance", headline))
    print(f"\nTotal Yahoo Finance Articles Fetched: {len(headlines)}\n")
    return headlines

# CNBC Scraper (Same as before)
def get_cnbc_articles():
    url = "https://www.cnbc.com/world/?region=world"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for item in soup.find_all("div", class_="LatestNews-container"):
        headline = item.find("a", class_="LatestNews-headline")
        timestamp = item.find("time")
        if headline and timestamp:
            timestamp_text = timestamp.text.strip()
            try:
                article_time = convert_relative_time(timestamp_text)
                if article_time.date() == datetime.now().date():
                    headlines.append(("CNBC", headline.text))
            except ValueError:
                continue
    print(f"\nTotal CNBC Articles Fetched: {len(headlines)}\n")
    return headlines

# Sentiment Analysis with Source Info
def analyze_sentiment(headlines):
    if not headlines:
        print("No headlines found for sentiment analysis.\n")
        return
    for source, headline in headlines:
        analysis = TextBlob(headline)
        polarity = analysis.sentiment.polarity
        sentiment = "Positive" if polarity > 0.3 else "Negative" if polarity < -0.3 else "Neutral"
        print(f"\nSource: {source}\nHeadline: {headline}\nSentiment: {sentiment} | Polarity: {polarity}\n")

# Main Execution
if __name__ == "__main__":
    print("Fetching today's Yahoo Finance articles...\n")
    yahoo_articles = get_yahoo_finance_articles()
    analyze_sentiment(yahoo_articles)

    print("\nFetching today's CNBC articles...\n")
    cnbc_articles = get_cnbc_articles()
    analyze_sentiment(cnbc_articles)
