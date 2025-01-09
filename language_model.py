import openai
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="API_key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Fetch article content
def get_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()[:10000]  # Limit to avoid token overflow

# Summarize article using OpenAI
def summarize_article(article_text):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"can you tell me if this is ether Fantastic, Good, Great, ,Neutral, or Bad for AAPL only respond with one word?: {article_text}",
        max_tokens=500
    )
    return response['choices'][0]['text']

# Example usage
article_url = "https://finance.yahoo.com/news/apple-nvidia-microsoft-amazon-alphabet-205532443.html"
article_text = get_article_text(article_url)
summary = summarize_article(article_text)
print("Summary:\n", summary)
