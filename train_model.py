import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load cleaned and labeled data
data = pd.read_csv("C:/Users/Lohitha/OneDrive/Desktop/fake_news_detector/fake-and-real-news-dataset/cleaned_data.csv")

# Split data into input (X) and target (y)
X = data['text']
y = data['label']

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model,"fake_news_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")