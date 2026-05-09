import re
import string
import pandas as pd

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text.strip()

def load_and_clean_data(path):
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    # Your dataset has 'text'
    text_col = 'text'  # after lowercasing 'Text' → 'text'
    df['clean_text'] = df[text_col].apply(clean_tweet)
    df = df[df['clean_text'].str.len() > 5]
    return df