import streamlit as st
import pandas as pd
import pickle
import os
from utils import preprocess_data, encrypt_user_id, log_access

# Load model and vectorizer
model = pickle.load(open('models/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Authentication
def check_password():
    """Simple password check for authentication."""
    def password_entered():
        if st.session_state["password"] == "secure123":  # Hardcoded for demo
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
    
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    return True

# Main app
def main():
    st.title("Sentiment Analysis App")
    
    if not check_password():
        st.error("Please enter the correct password.")
        return
    
    st.success("Authenticated! Welcome to the Sentiment Analysis App.")
    log_access("User authenticated and accessed app")
    
    # Check if users_comments.csv exists
    csv_path = os.path.join('data', 'users_comments.csv')
    if not os.path.exists(csv_path):
        st.error("Please provide a 'users_comments.csv' file in the 'data/' directory with 'user_id' and 'comment' columns.")
        return
    
    # Load inference dataset
    df = pd.read_csv(csv_path)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Predict sentiments
    X = vectorizer.transform(df['cleaned_text']).toarray()
    df['predicted_sentiment'] = model.predict(X)
    
    # Encrypt user IDs
    cipher = None
    df['encrypted_user_id'] = None
    for idx, row in df.iterrows():
        encrypted_id, cipher = encrypt_user_id(row['user_id'], cipher)
        df.at[idx, 'encrypted_user_id'] = encrypted_id.decode('utf-8')  # For display
    
    # Display all results
    st.subheader("All User Comments with Sentiment")
    st.dataframe(df[['encrypted_user_id', 'comment', 'predicted_sentiment']])
    
    log_access("All sentiment predictions accessed")

if __name__ == "__main__":
    main()