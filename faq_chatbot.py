import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample FAQs
faqs = {
    "what is your name?": "I am a simple FAQ chatbot.",
    "how can I reset my password?":
    "Click 'Forgot Password' on the login screen.",
    "what is your return policy?":
    "We allow returns within 30 days with a receipt.",
    "how to contact customer service?":
    "You can email us at support@example.com.",
    "do you ship internationally?":
    "Yes, we ship to over 50 countries worldwide."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Streamlit UI
st.title("🤖 FAQ Chatbot")
st.write("Ask a question and I'll try to help!")

user_input = st.text_input("Enter your question:")

if user_input:
    input_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(input_vec, question_vectors)
    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score > 0.5:
        st.success("Answer: " + answers[best_match_index])
    else:
        st.warning("Sorry, I don't understand the question.")
