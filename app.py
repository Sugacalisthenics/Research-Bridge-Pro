import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="Suga's Movie Insights", layout="wide")

# 2. Setup & Preprocessing
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')

setup_nltk()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
negation_words = {'not', 'no', 'never', 'nor', 'neither', 'but'}
all_stopwords = [word for word in all_stopwords if word not in negation_words]

def clean_review(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    return ' '.join(review)

# 3. Models Load Karne ka safe tareeka
@st.cache_resource
def load_models():
    # Sentiment Model aur Vectorizer load karein
    model = pickle.load(open('movie_sentiment_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    
    # Recommendation ke liye movies list load karein
    if os.path.exists('movies_list.pkl'):
        movies_df = pickle.load(open('movies_list.pkl', 'rb'))
    else:
        # Fallback agar file missing ho
        movies_df = pd.DataFrame({'title': ['Inception', 'Avatar', 'RRR', 'Interstellar', 'The Dark Knight']})
        
    return model, vectorizer, movies_df

# Load Models
try:
    model, cv, movies_df = load_models()
except Exception as e:
    st.error(f"Models load karne mein dikat hai: {e}")

# 4. Sidebar Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.selectbox("Menu", ["Sentiment Analysis", "Recommendations"])

# --- FEATURE 1: SENTIMENT ANALYSIS ---
if choice == "Sentiment Analysis":
    st.title("🎬 Movie Review Sentiment Analysis")
    user_input = st.text_area("Write your movie review here:", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            cleaned_text = clean_review(user_input)
            vectorized_text = cv.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            
            if prediction == 1:
                st.success("Review is POSITIVE 😊")
            else:
                st.error("Review is NEGATIVE 😞")
        else:
            st.warning("Kuch toh likho review mein!")

# --- FEATURE 2: RECOMMENDATIONS ---
elif choice == "Recommendations":
    st.title("🍿 Movie Recommendation System")
    st.subheader("Select a Movie you like")
    
    # Dropdown menu from movies_df
    selected_movie = st.selectbox("Choose a movie:", movies_df['title'].values)
    
    if st.button("Recommend"):
        st.write(f"Aapne chuni: **{selected_movie}**")
        st.subheader("Similar Movies for you:")
        
        # Dummy Recommendations (Jab tak aap similarity matrix load nahi karte)
        recommendations = ["The Dark Knight", "Inception", "Interstellar", "Avatar", "RRR"]
        
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                # Placeholder image
                st.image("https://via.placeholder.com/150", caption=recommendations[i])