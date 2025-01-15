import openai
import requests
from bs4 import BeautifulSoup
import os
import json
import csv
from dotenv import load_dotenv

load_dotenv(dotenv_path="API_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load companies from JSON file
def load_companies(json_file="companies.json"):
    with open(json_file, "r") as file:
        data = json.load(file)
        return data["companies"]

# Fetch article titles from Yahoo Finance and CNBC (last 24 hours)
def fetch_articles():
    urls = [
        "https://finance.yahoo.com/",
        "https://www.cnbc.com/world/?region=world"
    ]
    titles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        for item in soup.find_all("h3"):
            headline = item.get_text(strip=True)
            titles.append(headline)
    return titles

# GPT filters the top articles based on importance
def gpt_filter_articles(titles):
    prompt = "Here are some finance-related articles. Select the 10-15 most relevant for market trends:\n" + "\n".join(titles)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    selected_titles = response['choices'][0]['text'].splitlines()
    return [title.strip() for title in selected_titles if title.strip()]

# Fetch article content
def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()[:4000]  # Token limit for safety

# GPT summarizes the article and identifies affected companies
def summarize_and_identify_companies(article_text, companies):
    prompt = f"Summarize the following article and identify if any of these companies: {', '.join(companies)} are mentioned, along with whether the article is good, bad, or neutral for the companies mentioned: \n\n{article_text}"
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500
    )
    return response['choices'][0]['text']

# Initialize CSV only if it doesn't exist
def initialize_csv(companies, filename="stock_analysis.csv"):
    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            header = ["Company", "GPT Analysis", "GPT Standalone", "Linear Regression", "Neural Network"]
            writer.writerow(header)
            for company in companies:
                writer.writerow([company, "", "", "", ""])
        print(f"\n  CSV initialized with 'GPT Standalone' as the third column.")

#   Update CSV without Overwriting Existing Data
def update_csv_with_gpt_standalone(results, filename="stock_analysis.csv"):
    with open(filename, mode="r") as file:
        reader = list(csv.reader(file))
        header = reader[0]
        column_index = header.index("GPT Standalone")  # Specifically target the third column

    # Modify the specified column without overwriting other columns
    for row in reader[1:]:
        company = row[0]
        # Only update if the company result is provided, else retain old data
        if company in results:
            row[column_index] = results[company]

    # Save the modified CSV back without overwriting other columns
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(reader)
    print(f"\n  CSV successfully updated with 'GPT Standalone' results (Preserved other columns).")

# Main Execution
if __name__ == "__main__":
    # Load companies and ensure CSV exists without overwriting existing data
    companies = load_companies("companies.json")
    initialize_csv(companies)

    # Fetch articles and filter them using GPT
    print("\nFetching Articles...")
    articles = fetch_articles()
    selected_articles = gpt_filter_articles(articles)
    print(f"\nGPT Selected Articles: {selected_articles}")

    # Analyze the selected articles and summarize them
    print("\nAnalyzing selected articles...")
    gpt_results = {}
    for article in selected_articles:
        article_text = get_article_text("https://www.cnbc.com/world/?region=world")  # Replace with actual URLs if possible
        summary = summarize_and_identify_companies(article_text, companies)
        
        # Parse GPT output for company mentions
        for company in companies:
            if company in summary:
                sentiment = "Good" if "Good" in summary else "Bad" if "Bad" in summary else "Neutral"
                gpt_results[company] = sentiment

    #   Update only the "GPT Standalone" column while keeping all other columns intact
    update_csv_with_gpt_standalone(gpt_results)
    print("\n  Pipeline completed successfully! 'GPT Standalone' column updated while preserving other data.")
