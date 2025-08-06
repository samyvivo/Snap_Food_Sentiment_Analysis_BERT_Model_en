# Snap_Food_Sentiment_Analysis_BERT_Model_en

# 🍔 SnapFood Sentiment Analysis (BERT - Persian)

## English Description

### 📌 Project Overview
This project implements a **Persian Sentiment Analysis Model** for **SnapFood** user reviews using a **BERT-based Transformer** model from Hugging Face.  
It processes Persian restaurant and food delivery reviews to classify sentiments as **Positive** or **Negative**.

### 🛠 Features
- Preprocessing Persian text (removing non-Persian characters, cleaning spaces, normalization)
- Tokenization with Hugging Face `AutoTokenizer`
- Fine-tuned **BERT** model for Persian sentiment classification
- Evaluation with accuracy, precision, recall, F1-score
- Interactive **Streamlit** web app for real-time sentiment prediction

### 📂 Project Structure

📁 Snap_Food_Sentiment_Analysis_BERT_Model_fa
│── 📄 app.py # Streamlit app
│── 📄 model_loader.py # Model loading & preprocessing
│── 📄 requirements.txt # Dependencies
│── 📄 README.md # Project documentation
│── 📁 model/ # Saved fine-tuned BERT model
│── 📁 data/ # Dataset (SnapFood reviews)


### 🚀 How to Run Locally
#### 1️⃣ Clone Repository
```bash
git clone https://github.com/samyvivo/Snap_Food_Sentiment_Analysis_BERT_Model_fa.git
cd Snap_Food_Sentiment_Analysis_BERT_Model_fa

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

| Review (Persian)                       | Predicted Sentiment |
| -------------------------------------- | ------------------- |
| کیفیت پیتزا عالی بود و خیلی سریع رسید. | Positive ✅          |
| غذا دیر رسید و خیلی سرد بود.           | Negative ❌          |

🖼 Screenshot

<img width="1251" height="620" alt="image" src="https://github.com/user-attachments/assets/caf4c3f1-2205-4653-b4fa-83868d66384d" />
