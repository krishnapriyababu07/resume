import streamlit as st
import pandas as pd
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os

# Function to preprocess text
def preprocess_text_simple(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text()
    return text

# Absolute path to your dataset
file_path = 'clean_resume_data.csv'

# Initialize vectorizer and model
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
model = MultinomialNB()

# Load the dataset and train the model if the file exists
if os.path.exists(file_path):
    resume_data = pd.read_csv(file_path)
    resume_data['Feature'] = resume_data['Feature'].apply(preprocess_text_simple)
    
    X = vectorizer.fit_transform(resume_data['Feature'])
    y = resume_data['Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
else:
    accuracy = None

# Build the Streamlit app
st.title("Resume Categorization")

# File uploader
uploaded_file = st.file_uploader("Upload your resume (PDF format only)", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # Display the extracted text (optional)
    st.write("Extracted Text from PDF:")
    st.write(resume_text)

    # Process and predict
    resume_text_processed = preprocess_text_simple(resume_text)
    resume_vectorized = vectorizer.transform([resume_text_processed])
    
    # Predict the category
    prediction = model.predict(resume_vectorized) if accuracy is not None else "Model not trained"
    
    # Display the prediction
    st.write(f"Predicted Field: {prediction[0]}")
    
    # Display model accuracy
    if accuracy is not None:
        st.write(f"Model Accuracy: {accuracy:.2f}")
    else:
        st.write("Model is not trained.")

