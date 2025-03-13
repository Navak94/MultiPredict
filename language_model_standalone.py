import openai
import requests
import pandas as pd
import time
import json
from bs4 import BeautifulSoup
from newspaper import Article
from datetime import datetime, timedelta
from dateutil import parser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(dotenv_path="API_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define reputable sources
REPUTABLE_SOURCES = ["yahoo.com", "bloomberg.com", "cnbc.com", "reuters.com", "marketwatch.com"]

# Load company names
def load_companies(file_path="companies_names.txt"):
    companies = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                ticker, name = line.strip().split(": ", 1)
                companies[ticker] = name
    return companies

# Parse Google News dates
def parse_google_news_date(date_text):
    try:
        return parser.parse(date_text)
    except:
        return None

# Search Google News for stock articles within the last 14 days
def search_news(company_name):
    end_date = datetime.today().strftime("%m/%d/%Y")  # Today's date
    start_date = (datetime.today() - timedelta(days=14)).strftime("%m/%d/%Y")  # 14 days ago

    search_url = (
        f"https://www.google.com/search?q={company_name.replace(' ', '+')}+stock+news&tbm=nws"
        f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"  # Custom Date Range
    )
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for result in soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd"):
        title = result.get_text()
        parent = result.find_parent("a")
        if parent and "href" in parent.attrs:
            link = parent["href"]
            if link.startswith("/url?q="):
                link = link[7:].split("&")[0]  # Extract clean URL
            articles.append((title, link))

    return filter_reputable_sources(articles[:15])

# Filter only reputable sources
def filter_reputable_sources(articles):
    return [(title, link) for title, link in articles if any(source in link for source in REPUTABLE_SOURCES)]

# Robust OpenAI call with retry logic
def robust_openai_call(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=300
            )
            return response["choices"][0]["text"].strip()
        except openai.error.OpenAIError as e:
            print(f"⚠️ OpenAI API error: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return "Error: Failed to generate response"

# Fetch article content using newspaper3k
def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text[:5000]  # Limit to 5000 characters
    except:
        return ""

# GPT selects the best articles
def gpt_select_articles(articles, company):
    titles = [title for title, _ in articles]
    prompt = f"""
    Here are recent news articles about {company}. Select the 5 most relevant for stock trends:
    {chr(10).join(titles)}
    Respond with just the selected titles, one per line.
    """
    response_text = robust_openai_call(prompt)
    selected_titles = response_text.split("\n")
    return [article for article in articles if article[0] in selected_titles]

# GPT summarizes articles
def gpt_summarize_article(article_text, company):
    prompt = f"""
    Summarize this article and classify its impact on {company}'s stock as Good, Neutral, or Bad.
    {article_text}
    """
    return robust_openai_call(prompt)

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

# Main function to run GPT-based sentiment analysis
def gpt_sentiment_analysis():
    companies = load_companies("companies_names.txt")
    results = []
    sentiment_aggregate = {company: [] for company in companies.values()}

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

        sentiment_aggregate[company] = company_sentiments
        time.sleep(2)  # Avoid hitting API limits

    df = pd.DataFrame(results, columns=["Ticker", "Company", "Title", "URL", "Sentiment", "Summary"])
    df.to_csv("GPT_standalonecompany_sentiment_analysis.csv", index=False)
    print("\n✅ Sentiment analysis completed! Results saved to GPT_standalonecompany_sentiment_analysis.csv")

    create_summary_csv(sentiment_aggregate, companies)

# Run the analysis
gpt_sentiment_analysis()
