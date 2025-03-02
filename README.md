# Fake-News-Detection-
News,Label
"Government launches new tax reforms",Real
"Scientists discover a cure for baldness",Real
"Aliens spotted in New York City",Fake
"Secret government project leaks online",Fake
"New AI technology to replace doctors",Real
"Man claims he can time travel",Fake
"Famous celebrity admits to being a robot",Fake
"Doctors recommend new health diet",Real
"Scientists confirm Earth is actually flat",Fake
"NASA plans mission to Mars in 2026",Real

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv("fake_news.csv")

# Split features and labels
X = data["News"]
y = data["Label"]

# Convert text into numerical form
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict for test data
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict new headlines
new_headlines = ["Aliens have landed on Earth", "NASA discovers a new planet"]
new_headlines_vectorized = vectorizer.transform(new_headlines)
predictions = model.predict(new_headlines_vectorized)

# Display results
for news, label in zip(new_headlines, predictions):
    print(f"News: {news} → Prediction: {label}")

    # Fake News Detection Using Machine Learning

This project uses a simple Naive Bayes classifier to classify news headlines as **Real** or **Fake**.

## Requirements
- Python
- Pandas
- Scikit-learn

## How to Run
1. Install dependencies: `pip install pandas scikit-learn`
2. Run the script: `python fake_news_detection.py`

## Example Prediction
Input: `"Aliens have landed on Earth"`  
Output: **Fake**
