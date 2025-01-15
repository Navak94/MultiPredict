import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import csv
from datetime import datetime, timedelta

# Load companies from the provided companies_names.txt file
def load_companies(filename="companies_names.txt"):
    companies = {}
    with open(filename, "r") as file:
        for line in file:
            try:
                ticker, name = line.strip().split(": ")
                companies[ticker.strip()] = name.strip()
            except ValueError:
                continue
    return companies

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

# Scrape Yahoo Finance articles with links
def get_yahoo_finance_articles():
    url = "https://finance.yahoo.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for item in soup.find_all("h3"):
        headline = item.get_text(strip=True)
        link = item.find_parent("a")["href"] if item.find_parent("a") else url
        full_link = f"https://finance.yahoo.com{link}" if link.startswith("/") else link
        articles.append(("Yahoo Finance", headline, full_link))
    print(f"\n✅ Total Yahoo Finance Articles Fetched: {len(articles)}")
    return articles

# Scrape CNBC articles with links
def get_cnbc_articles():
    url = "https://www.cnbc.com/world/?region=world"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    for item in soup.find_all("div", class_="LatestNews-container"):
        headline = item.find("a", class_="LatestNews-headline")
        timestamp = item.find("time")
        if headline and timestamp:
            link = headline["href"]
            full_link = f"https://www.cnbc.com{link}" if link.startswith("/") else link
            try:
                article_time = convert_relative_time(timestamp.text.strip())
                if article_time.date() == datetime.now().date():
                    articles.append(("CNBC", headline.text, full_link))
            except ValueError:
                continue
    print(f"\n✅ Total CNBC Articles Fetched: {len(articles)}")
    return articles

# Sentiment analysis with article links and mention tracking
def analyze_sentiment_and_companies(articles, companies):
    sentiment_results = {ticker: {"mentions": 0, "sentiments": [], "links": []} for ticker in companies}

    for source, headline, link in articles:
        analysis = TextBlob(headline)
        polarity = analysis.sentiment.polarity
        sentiment = "Positive" if polarity > 0.3 else "Negative" if polarity < -0.3 else "Neutral"

        # Check for company mentions and count them
        for ticker, company_name in companies.items():
            if ticker in headline or company_name in headline:
                sentiment_results[ticker]["mentions"] += 1
                sentiment_results[ticker]["sentiments"].append(polarity)
                sentiment_results[ticker]["links"].append(link)

    # Calculate the average sentiment for each company mentioned
    averaged_results = {}
    for ticker, data in sentiment_results.items():
        if data["mentions"] > 0:
            avg_score = sum(data["sentiments"]) / data["mentions"]
            avg_sentiment = "Positive" if avg_score > 0.3 else "Negative" if avg_score < -0.3 else "Neutral"
            averaged_results[ticker] = {
                "sentiment": avg_sentiment, 
                "mentions": data["mentions"], 
                "links": "; ".join(data["links"])
            }
        else:
            averaged_results[ticker] = {"sentiment": "Not Mentioned", "mentions": 0, "links": ""}
    return averaged_results

# Initialize CSV with links included
def initialize_csv(companies, filename="sentiment_analysis.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Company", "Company Name", "Sentiment Analysis", "Total Mentions", "Source Links"])
        for ticker, name in companies.items():
            writer.writerow([ticker, name, "Not Mentioned", 0, ""])
    print(f"\n✅ CSV initialized with 'Source Links' column included.")

# Update CSV with sentiment results and article links
def update_csv_with_sentiment_results(results, filename="sentiment_analysis.csv"):
    with open(filename, mode="r") as file:
        reader = list(csv.reader(file))
        header = reader[0]

    for row in reader[1:]:
        ticker = row[0]
        if ticker in results:
            row[2] = results[ticker]["sentiment"]
            row[3] = str(results[ticker]["mentions"])
            row[4] = results[ticker]["links"]

    # Save back to CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(reader)
    print(f"\n✅ CSV successfully updated with links and sentiment results!")

# Main Execution
if __name__ == "__main__":
    # Load companies and initialize CSV
    companies = load_companies("companies_names.txt")
    initialize_csv(companies)

    # Fetch articles from Yahoo Finance and CNBC
    print("Fetching Articles...")
    yahoo_articles = get_yahoo_finance_articles()
    cnbc_articles = get_cnbc_articles()
    all_articles = yahoo_articles + cnbc_articles

    # Perform sentiment analysis and link tracking
    print("\nAnalyzing Sentiment...")
    sentiment_results = analyze_sentiment_and_companies(all_articles, companies)

    # Update the CSV with sentiment results and links
    update_csv_with_sentiment_results(sentiment_results)
    print("\n✅ Sentiment Analysis Completed Successfully!")
