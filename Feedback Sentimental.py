import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Sample feedback data - replace with your actual dataset
data = {
    'intern_id': [101, 102, 103, 104],
    'feedback': [
        "I really enjoyed the internship, learned a lot!",
        "The tasks were okay, but sometimes felt repetitive.",
        "I didn't like the lack of guidance and communication.",
        "Overall, a decent experience with some challenges."
    ]
}

df = pd.DataFrame(data)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment based on compound score
def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment classification
df['sentiment'] = df['feedback'].apply(classify_sentiment)

print(df[['intern_id', 'feedback', 'sentiment']])
