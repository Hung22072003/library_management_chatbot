import json
import numpy as np
import nltk
import os
import pandas as pd
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.punkt import PunktLanguageVars
from nltk.data import find

# Cấu hình thư mục lưu trữ tài nguyên NLTK
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

# Danh sách các tài nguyên cần thiết
required_resources = [
    "punkt",
    'punkt_tab',
    "wordnet",
    "stopwords",
    "averaged_perceptron_tagger",
    "maxent_ne_chunker",
    "words"
]

# Tải tài nguyên nếu chưa có
for resource in required_resources:
    try:
        find(f"{'tokenizers' if resource == 'punkt' else 'corpora'}/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DIR)

# Khởi tạo tokenizer và lemmatizer
punkt_tokenizer = PunktLanguageVars()
lemmatizer = WordNetLemmatizer()

# Load mô hình
model = load_model("chatbot/data/chatbot_model.h5")

# Load intents
with open("chatbot/data/intents.json", encoding="utf-8") as f:
    intents = json.load(f)

# Tiền xử lý tập từ và lớp
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
