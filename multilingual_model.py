from transformers import pipeline

def load_model():
    model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    framework="pt"
)
    return model

def predict_sentiment(model, text):
    result = model(text)[0]
    return result['label'], result['score']
