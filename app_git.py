import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import zipfile
import io

# ------------------------
# Load your model & tokenizer
# ------------------------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/samyhusy/Snap_food_Sentiment_analysis/resolve/main/snapfood_model.zip"
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("model_dir")

    MODEL_PATH = "model_dir/snapfood_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------
# Sentiment Analysis Function
# ------------------------
def sentiment_analysis(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return predicted_class, confidence

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="SnapFood Sentiment Analysis", page_icon="🍔", layout="centered")

#st.title("🍔 SnapFood Sentiment Analysis (BERT)")
st.image("snapfood_logo.png", width=200)
st.markdown("### Sentiment Analysis (BERT Model)")
st.write("Enter a review in Persian and get sentiment prediction.")

# Text input
user_input = st.text_area("📝 Enter your review:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        label, confidence = sentiment_analysis(user_input)
        sentiment = "😊 Positive" if label == 1 else "😠 Negative"
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence:.2%}")
