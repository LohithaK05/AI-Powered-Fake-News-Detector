import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load cleaned dataset
data = pd.read_csv("C:/Users/Lohitha/OneDrive/Desktop/fake_news_detector/fake-and-real-news-dataset/cleaned_data.csv")

# Transform the text column
X = vectorizer.transform(data['text'])

# Predict
data['predicted_label'] = model.predict(X)

# Display results in the app
st.title("🧠 Fake News Detector Results on Dataset")

st.write("### Dataset Preview with Predictions")
st.dataframe(data[['title', 'label', 'predicted_label']].head(50))  # Limit to 50 for performance

# Optional: Show accuracy
accuracy = (data['label'] == data['predicted_label']).mean() * 100
st.write(f"✅ Model Accuracy on This Dataset: **{accuracy:.2f}%**")

# Optionally allow download
csv = data.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Full Results as CSV", csv, "predicted_results.csv", "text/csv")