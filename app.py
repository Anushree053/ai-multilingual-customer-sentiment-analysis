import streamlit as st
import pandas as pd
from multilingual_model import load_model, predict_sentiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from googletrans import Translator
translator = Translator()
# Load model
model = load_model()

st.title("AI-Powered Multilingual Customer Sentiment & Feedback Intelligence System")
st.write("Supports English, Hindi, Kannada, Spanish and more languages")

# ------------------ SINGLE INPUT ------------------
st.header("Single Feedback Analysis")

text = st.text_area("Enter Customer Feedback")

if st.button("Analyze Sentiment"):
    if text.strip() != "":
        translated = translator.translate(text, dest='en').text
        label, score = predict_sentiment(model, translated)

        if "4" in label or "5" in label:
            sentiment = "Positive"
        elif "3" in label:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2f}")
    else:
        st.warning("Please enter feedback")

# ------------------ CSV INPUT ------------------
st.header("Bulk CSV Analysis")

file = st.file_uploader("Upload CSV file)", type=["csv"])

def evaluate_model(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return accuracy, precision, recall, f1

if file is not None:
    df = pd.read_csv(file)

    # 🔥 FIX: Make all column names lowercase
    df.columns = df.columns.str.lower()

    if "feedback" not in df.columns:
        st.error("CSV must contain 'feedback' column")
    else:
        sentiments = []
        scores = []

        for t in df['feedback']:
            translated = translator.translate(text, dest='en').text
            label, score = predict_sentiment(model, translated)
            if "4" in label or "5" in label:
                sentiment = "positive"
            elif "3" in label:
                sentiment = "neutral"
            else:
                sentiment = "negative"
           

            sentiments.append(sentiment)
            scores.append(score)

        df['predicted_sentiment'] = sentiments
        df['confidence'] = scores

        st.write(df)

        # -------- METRICS --------
        if "sentiment" in df.columns:
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

            true_labels = df['sentiment'].str.lower().map(label_map)
            predicted_labels = df['predicted_sentiment'].str.lower().map(label_map)

            accuracy, precision, recall, f1 = evaluate_model(true_labels, predicted_labels)

            st.subheader("Model Evaluation Metrics")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")
        else:
            st.warning("Upload CSV with 'sentiment' column to see metrics")

        st.download_button(
            "Download Result",
            df.to_csv(index=False),
            "sentiment_output.csv",
            "text/csv"
        )