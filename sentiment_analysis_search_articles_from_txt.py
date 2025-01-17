import requests
import pandas as pd
import time
import json
from bs4 import BeautifulSoup
from textblob import TextBlob
import numpy as np

# Load company names from the provided text file
def load_companies(file_path="companies_names.txt"):
    companies = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                ticker, name = line.strip().split(": ", 1)
                companies[ticker] = name
    return companies

# Function to search Google News and get recent articles for a company
def search_news(company_name):
    search_url = f"https://www.google.com/search?q={company_name.replace(' ', '+')}+stock+news&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for item in soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd"):
        title = item.get_text()
        link = item.find_parent("a")["href"]
        if link.startswith("/url?q="):
            link = link[7:].split("&")[0]  # Extract clean URL
        articles.append((title, link))

    return articles[:5]  # Limit to top 5 articles per company

# Function to analyze sentiment of an article
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.2:
        return "Positive", sentiment
    elif sentiment < -0.2:
        return "Negative", sentiment
    else:
        return "Neutral", sentiment

# Fetch article content
def get_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:5000]  # Limit to 5000 characters
    except:
        return ""

# Classify sentiment score into "Good", "Neutral", "Bad"
def classify_sentiment(score):
    if score is None:
        return "No Data"
    elif score > 0.2:
        return "Good"
    elif score < -0.2:
        return "Bad"
    else:
        return "Neutral"

# Main function to fetch news, analyze sentiment, and save to CSV
def sentiment_analysis():
    companies = load_companies("companies_names.txt")
    results = []
    sentiment_scores = {}

    for ticker, company in companies.items():
        print(f"🔍 Fetching news for {company} ({ticker})...")
        articles = search_news(company)
        
        if not articles:
            print(f"❌ No news found for {company}. Skipping...")
            continue

        company_sentiments = []

        for title, url in articles:
            print(f"   - Analyzing: {title}")
            article_text = get_article_text(url)
            if not article_text:
                print("     ❌ Could not fetch article text. Skipping...")
                continue

            sentiment_label, sentiment_score = analyze_sentiment(article_text)
            company_sentiments.append(sentiment_score)
            results.append([ticker, company, title, url, sentiment_label, sentiment_score])

        # Store all sentiment scores per company for later averaging
        if company_sentiments:
            sentiment_scores[company] = company_sentiments
        else:
            sentiment_scores[company] = []  # No data available

        time.sleep(2)  # Avoid sending too many requests too quickly

    # Save article sentiment results to CSV
    df = pd.DataFrame(results, columns=["Ticker", "Company", "Title", "URL", "Sentiment", "Score"])
    df.to_csv("company_sentiment_analysis.csv", index=False)
    print("\n✅ Sentiment analysis completed! Results saved to company_sentiment_analysis.csv")

    # Create and save the summary CSV with averages & medians
    create_summary_csv(sentiment_scores)

# Function to create company sentiment summary CSV
def create_summary_csv(sentiment_scores):
    summary_data = []

    for company, scores in sentiment_scores.items():
        if scores:
            avg_sentiment = np.mean(scores)
            median_sentiment = np.median(scores)
        else:
            avg_sentiment = None
            median_sentiment = None
        
        # Classify average and median sentiment into categories
        avg_label = classify_sentiment(avg_sentiment)
        median_label = classify_sentiment(median_sentiment)

        summary_data.append([company, avg_sentiment, avg_label, median_sentiment, median_label])

    summary_df = pd.DataFrame(summary_data, columns=["Company", "Average Sentiment Score", "Average Sentiment Label", "Median Sentiment Score", "Median Sentiment Label"])
    summary_df.to_csv("company_sentiment_summary.csv", index=False)
    print("\n✅ Summary CSV created: company_sentiment_summary.csv")

# Run the analysis
sentiment_analysis()
