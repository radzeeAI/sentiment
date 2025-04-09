import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from cryptography.fernet import Fernet
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')  # Added this to fix the error
nltk.download('stopwords')
nltk.download('wordnet')

# Ensure logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup logging
logging.basicConfig(filename=os.path.join(log_dir, 'log_access.txt'), level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def log_access(action):
    """Log an action to the file."""
    logging.info(action)

def clean_text(text):
    """Clean text: lowercase, remove stop words, and lemmatize."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
    return ' '.join(cleaned)

def encrypt_user_id(user_id, cipher=None):
    """Encrypt user ID."""
    if cipher is None:
        key = Fernet.generate_key()
        cipher = Fernet(key)
    encrypted_id = cipher.encrypt(str(user_id).encode())
    return encrypted_id, cipher

def decrypt_user_id(encrypted_id, cipher):
    """Decrypt user ID."""
    return cipher.decrypt(encrypted_id).decode()

def preprocess_data(df, text_column='comment'):
    """Preprocess dataset for modeling or inference."""
    df['cleaned_text'] = df[text_column].apply(clean_text)
    return df