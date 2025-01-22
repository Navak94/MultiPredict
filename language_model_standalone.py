import openai
import requests
import pandas as pd
import time
import json
from bs4 import BeautifulSoup

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="API_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load company names
def load_companies(file_path="companies_names.txt"):
    companies = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                ticker, name = line.strip().split(": ", 1)
                companies[ticker] = name
    return companies

# Search Google News for stock articles
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

    return articles[:15]  # Fetch up to 15 articles per company

# GPT picks the best articles
def gpt_select_articles(articles, company):
    titles = [title for title, _ in articles]
    prompt = f"""
    Here are recent news articles about {company}. Select the 5 most relevant for stock trends:
    {chr(10).join(titles)}
    Respond with just the selected titles, one per line.
    """
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300
    )
    selected_titles = response["choices"][0]["text"].strip().split("\n")
    return [article for article in articles if article[0] in selected_titles]

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

# GPT summarizes the articles and assigns sentiment
def gpt_summarize_article(article_text, company):
    prompt = f"""
    Summarize this article and classify its impact on {company}'s stock as Good, Neutral, or Bad.
    {article_text}
    """
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300
    )
    return response["choices"][0]["text"].strip()

# Main function to run GPT-based sentiment analysis
def gpt_sentiment_analysis():
    companies = load_companies("companies_names.txt")
    results = []
    sentiment_aggregate = {company: [] for company in companies.values()}  # Ensure all companies are included

    for ticker, company in companies.items():
        print(f"🔍 Searching news for {company} ({ticker})...")
        articles = search_news(company)

        if not articles:
            print(f"❌ No news found for {company}. Recording as 'No Data'...")
            sentiment_aggregate[company] = []
            continue

        selected_articles = gpt_select_articles(articles, company)

        if not selected_articles:
            print(f"❌ No relevant article selected for {company}. Recording as 'No Data'...")
            sentiment_aggregate[company] = []
            continue

        company_sentiments = []
        
        for title, url in selected_articles:
            print(f"   - Analyzing: {title}")
            article_text = get_article_text(url)
            if not article_text:
                print("     ❌ Could not fetch article text. Skipping...")
                continue

            summary = gpt_summarize_article(article_text, company)
            sentiment = "Good" if "Good" in summary else "Bad" if "Bad" in summary else "Neutral"
            results.append([ticker, company, title, url, sentiment, summary])
            company_sentiments.append(sentiment)

        # Store overall sentiment for the company
        sentiment_aggregate[company] = company_sentiments
        time.sleep(2)  # Avoid hitting API limits

    # Save article sentiment results to CSV
    df = pd.DataFrame(results, columns=["Ticker", "Company", "Title", "URL", "Sentiment", "Summary"])
    df.to_csv("GPT_standalonecompany_sentiment_analysis.csv", index=False)
    print("\n✅ Sentiment analysis completed! Results saved to GPT_standalonecompany_sentiment_analysis.csv")

    # Create and save summary CSV
    create_summary_csv(sentiment_aggregate, companies)

# Function to create a summary CSV with overall sentiment per company
def create_summary_csv(sentiment_aggregate, companies):
    summary_data = []
    
    for ticker, company in companies.items():
        sentiments = sentiment_aggregate.get(company, [])
        num_articles = len(sentiments)
        num_good = sentiments.count("Good")
        num_neutral = sentiments.count("Neutral")
        num_bad = sentiments.count("Bad")
        
        if sentiments:
            avg_sentiment = num_good - num_bad
            overall_sentiment = "Good" if avg_sentiment > 0 else "Bad" if avg_sentiment < 0 else "Neutral"
        else:
            overall_sentiment = "No Data"

        summary_data.append([ticker, company, overall_sentiment, num_articles, num_good, num_neutral, num_bad])
    
    summary_df = pd.DataFrame(summary_data, columns=["Ticker", "Company", "Overall Sentiment", "Number of Articles", "Good", "Neutral", "Bad"])
    summary_df.to_csv("GPT_standalone_company_sentiment_summary.csv", index=False)
    print("\n✅ Summary CSV created: GPT_standalonecompany_sentiment_summary.csv")

# Run the analysis
gpt_sentiment_analysis()
