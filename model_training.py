import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import joblib

from utils import load_and_clean_data

EDA_PATH = "static/eda/"
EVAL_PATH = "static/eval/"
MODEL_PATH = "static/model/"

os.makedirs(EDA_PATH, exist_ok=True)
os.makedirs(EVAL_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


def perform_eda(df):

    # -------------------------------
    # 1. Tweet Length Distribution
    # -------------------------------
    df['length'] = df['clean_text'].apply(len)

    plt.figure()
    sns.histplot(df['length'], bins=50, kde=True)
    plt.title("Tweet Length Distribution")
    plt.xlabel("Length of Tweet")
    plt.ylabel("Frequency")
    plt.savefig(EDA_PATH + "1_length_distribution.png")
    plt.close()

    # -------------------------------
    # 2. Word Count Distribution
    # -------------------------------
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    plt.figure()
    sns.histplot(df['word_count'], bins=50, kde=True)
    plt.title("Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.savefig(EDA_PATH + "2_wordcount_distribution.png")
    plt.close()

    # -------------------------------
    # 3. Top 20 Most Common Words
    # -------------------------------
    all_words = " ".join(df['clean_text']).split()
    common_words = Counter(all_words).most_common(20)

    words = [w[0] for w in common_words]
    counts = [w[1] for w in common_words]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words)
    plt.title("Top 20 Most Frequent Words")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.savefig(EDA_PATH + "3_top_words.png")
    plt.close()

    # -------------------------------
    # 4. Sentiment Distribution (after VADER)
    # -------------------------------
    if 'sentiment' in df.columns:
        plt.figure()
        sns.countplot(x='sentiment', data=df)
        plt.title("Sentiment Distribution")
        plt.savefig(EDA_PATH + "4_sentiment_distribution.png")
        plt.close()

    # -------------------------------
    # 5. Word Count vs Sentiment
    # -------------------------------
    if 'sentiment' in df.columns:
        plt.figure()
        sns.boxplot(x='sentiment', y='word_count', data=df)
        plt.title("Word Count vs Sentiment")
        plt.savefig(EDA_PATH + "5_wordcount_vs_sentiment.png")
        plt.close()

    # -------------------------------
    # 6. Tweet Length vs Sentiment
    # -------------------------------
    if 'sentiment' in df.columns:
        plt.figure()
        sns.boxplot(x='sentiment', y='length', data=df)
        plt.title("Tweet Length vs Sentiment")
        plt.savefig(EDA_PATH + "6_length_vs_sentiment.png")
        plt.close()

    # -------------------------------
    # 7. Top Languages Distribution
    # -------------------------------
    if 'lang' in df.columns:
        plt.figure()
        df['lang'].value_counts().head(10).plot(kind='bar')
        plt.title("Top Languages Used")
        plt.xlabel("Language")
        plt.ylabel("Count")
        plt.savefig(EDA_PATH + "7_language_distribution.png")
        plt.close()

    # -------------------------------
    # 8. Retweet Count Distribution
    # -------------------------------
    if 'retweet_count' in df.columns:
        plt.figure()
        sns.histplot(df['retweet_count'], bins=50)
        plt.title("Retweet Count Distribution")
        plt.savefig(EDA_PATH + "8_retweet_distribution.png")
        plt.close()

    # -------------------------------
    # 9. Favorites vs Retweets
    # -------------------------------
    if 'favorite_count' in df.columns and 'retweet_count' in df.columns:
        plt.figure()
        sns.scatterplot(x='retweet_count', y='favorite_count', data=df)
        plt.title("Retweets vs Favorites")
        plt.savefig(EDA_PATH + "9_retweet_vs_favorite.png")
        plt.close()


def apply_vader(df):
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df['sentiment'] = df['clean_text'].apply(get_sentiment)
    df.to_csv("static/model/labeled_data.csv", index=False)
    return df


def train_models(df):
    X = df['clean_text']
    y = df['sentiment']

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ 5+ Algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = []

    best_model = None
    best_score = 0
    best_name = ""

    # -------------------------------
    # Train + Evaluate
    # -------------------------------
    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results.append((name, acc))

        # -------------------------------
        # Save Classification Report
        # -------------------------------
        report = classification_report(y_test, preds)
        with open(EVAL_PATH + f"{name}_report.txt", "w") as f:
            f.write(report)

        # -------------------------------
        # Save Confusion Matrix
        # -------------------------------
        cm = confusion_matrix(y_test, preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(EVAL_PATH + f"{name}_cm.png")
        plt.close()

        # Track best model
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    # -------------------------------
    # Accuracy Comparison Graph
    # -------------------------------
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Accuracy", y="Model", data=results_df)
    plt.title("Model Accuracy Comparison")
    plt.xlim(0, 1)
    plt.savefig(EVAL_PATH + "accuracy_comparison.png")
    plt.close()

    # -------------------------------
    # Save Best Model
    # -------------------------------
    joblib.dump(best_model, MODEL_PATH + "best_model.pkl")
    joblib.dump(vectorizer, MODEL_PATH + "vectorizer.pkl")

    with open(MODEL_PATH + "metrics.txt", "w") as f:
        f.write(f"Best Model: {best_name}\nAccuracy: {best_score:.4f}")

    print(f"Best Model: {best_name} with Accuracy: {best_score:.4f}")

    return best_name