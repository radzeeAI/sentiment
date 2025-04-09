# Sentiment Analysis App

This project implements a secure sentiment analysis pipeline with a Streamlit frontend.

## Features
- **Authentication**: Password-protected access (default: `secure123`).
- **Training Data**: Hardcoded simple dataset for model training.
- **Inference Data**: `users_comments.csv` for sentiment analysis (sample created if not provided).
- **Sentiment Analysis**: Trained Logistic Regression model predicts sentiment.
- **Security**: User IDs encrypted for negative sentiment comments.
- **Logging**: Access logged to `logs/log_access.txt`.

## Setup
1. Clone the repo: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train_model.py`
4. Run the app: `streamlit run app.py`

## Structure
- `data/`: Inference dataset (`users_comments.csv`)
- `models/`: Trained model and vectorizer
- `logs/`: Access logs
- `app.py`: Streamlit app
- `utils.py`: Helper functions
- `train_model.py`: Model training with hardcoded data