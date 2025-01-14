from textblob import TextBlob

text = "Stock market today: Dow pops higher, tech weighs on Nasdaq as Treasury yields keep climbing"
analysis = TextBlob(text)
print("Sentiment Polarity:", analysis.sentiment.polarity)  # Outputs a positive score
