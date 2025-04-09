import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from utils import preprocess_data, log_access
import os

def create_training_data():
    """Create a simple hardcoded dataset for training."""
    data = {
        'text': [
            'I love this product',
            'This is terrible',
            'Amazing experience',
            'Not good at all',
            'Really happy with it'
        ],
        'sentiment': [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative
    }
    df = pd.DataFrame(data)
    return df

def train_and_save_model():
    """Train a sentiment model on hardcoded data and save it."""
    df = create_training_data()
    df = preprocess_data(df, text_column='text')
    
    # Transform text
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['sentiment']
    
    # Train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Ensure models directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save model and vectorizer
    with open(os.path.join(models_dir, 'sentiment_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    log_access("Model trained and saved")
    print("Model trained and saved!")

if __name__ == "__main__":
    train_and_save_model()