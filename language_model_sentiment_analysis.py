import openai
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import os
import json
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(dotenv_path="API_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load companies from JSON
def load_companies(json_file="companies.json"):
    with open(json_file, "r") as file:
        data = json.load(file)
        return data["companies"]

# Convert relative timestamps from CNBC
def convert_relative_time(relative_time):
    if "Min Ago" in relative_time:
        minutes = int(relative_time.split(" ")[0])
        return datetime.now() - timedelta(minutes=minutes)
    elif "Hours Ago" in relative_time:
        hours = int(relative_time.split(" ")[0])
        return datetime.now() - timedelta(hours=hours)
    else:
        return datetime.now()

# Scrape Yahoo Finance articles with URLs
def get_yahoo_finance_articles():
    url = "https://finance.yahoo.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for item in soup.find_all("a", href=True):
        headline = item.get_text(strip=True)
        link = item['href']
        if headline and link.startswith("https://finance.yahoo.com"):
            articles.append({"source": "Yahoo Finance", "headline": headline, "url": link})
    return articles

# Scrape CNBC articles with URLs
def get_cnbc_articles():
    url = "https://www.cnbc.com/world/?region=world"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for item in soup.find_all("a", class_="LatestNews-headline", href=True):
        headline = item.get_text(strip=True)
        link = item['href']
        timestamp = item.find_next("time")
        if timestamp and timestamp.text:
            article_time = convert_relative_time(timestamp.text.strip())
            if article_time.date() == datetime.now().date():
                articles.append({"source": "CNBC", "headline": headline, "url": link})
    return articles

# Perform sentiment filtering to reduce unnecessary API calls
def filter_by_sentiment(articles):
    filtered_articles = []
    for article in articles:
        analysis = TextBlob(article["headline"])
        polarity = analysis.sentiment.polarity
        sentiment = "Positive" if polarity > 0.3 else "Negative" if polarity < -0.3 else "Neutral"
        article["sentiment"] = sentiment
        # Keep only articles with strong sentiment
        if sentiment != "Neutral":
            filtered_articles.append(article)
    return filtered_articles

# Let GPT pick the top articles for deeper analysis
def gpt_filter_articles(articles):
    titles = [f"{article['headline']} - {article['url']}" for article in articles]
    prompt = f"Here are some finance-related articles with sentiment analysis. Select the 10-15 most important ones for further review:\n" + "\n".join(titles)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    selected_titles = response['choices'][0]['text'].splitlines()
    selected_articles = [article for article in articles if any(title in article["headline"] for title in selected_titles)]
    return selected_articles

# Fetch article content with length control
def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    full_text = soup.get_text()
    if len(full_text) > 4000:
        return full_text[:4000]  # Truncate large articles
    return full_text

# Splitting long articles only when needed
def split_text_into_chunks(text, chunk_size=3000):
    if len(text) <= chunk_size:
        return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Use GPT for summarization and company identification
def summarize_and_identify_companies(article_text, companies):
    # Batching only if necessary
    chunks = split_text_into_chunks(article_text)
    combined_summary = ""
    for chunk in chunks:
        prompt = f"Summarize the following article chunk and identify if any of these companies: {', '.join(companies)} are mentioned. If mentioned, determine if the impact is good, bad, or neutral: \n\n{chunk}"
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300  # Reduced to avoid overflow
        )
        combined_summary += response['choices'][0]['text'] + "\n"
    return combined_summary

# Initialize the CSV for recording analysis results
def initialize_csv(companies, filename="stock_analysis.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Company", "GPT Analysis", "Linear Regression", "Neural Network"]
        writer.writerow(header)
        for company in companies:
            writer.writerow([company, "", "", ""])
    print(f"\nCSV initialized with {len(companies)} companies.")

# Update CSV with results
def update_csv_with_gpt_results(results, filename="stock_analysis.csv"):
    with open(filename, mode="r") as file:
        reader = list(csv.reader(file))
    
    # Modify CSV by matching companies and updating the "GPT Analysis" column
    for company, sentiment in results.items():
        for row in reader:
            if row[0] == company:
                row[1] = sentiment  # Update GPT column

    # Save the modified CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(reader)
    print(f"\nCSV updated with GPT results.")

# Main Execution
if __name__ == "__main__":
    # Load companies and initialize the CSV
    companies = load_companies("companies.json")
    initialize_csv(companies)

    # Fetch articles and URLs
    yahoo_articles = get_yahoo_finance_articles()
    cnbc_articles = get_cnbc_articles()
    all_articles = yahoo_articles + cnbc_articles

    # Step 1: Sentiment Filtering
    sentiment_filtered_articles = filter_by_sentiment(all_articles)

    # Step 2: Ask GPT to pick top articles
    selected_articles = gpt_filter_articles(sentiment_filtered_articles)
    print(f"\nGPT Selected {len(selected_articles)} articles for deeper analysis.")

    # Step 3: Analyze articles and identify affected companies
    gpt_results = {}
    for article in selected_articles:
        article_text = get_article_text(article["url"])
        summary = summarize_and_identify_companies(article_text, companies)
        
        # Check for companies mentioned and their sentiment
        for company in companies:
            if company in summary:
                sentiment = "Good" if "Good" in summary else "Bad" if "Bad" in summary else "Neutral"
                gpt_results[company] = sentiment

    # Step 4: Update the CSV with results
    update_csv_with_gpt_results(gpt_results)
    print(" Results saved to 'stock_analysis.csv'.")
