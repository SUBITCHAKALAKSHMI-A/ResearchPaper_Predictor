import streamlit as st
import pickle
import pdfplumber  # To read PDF files
import pandas as pd


# Load the model
@st.cache_resource
def load_model():
    with open('sbert_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to evaluate the extracted text using the model
def evaluate_text(model, text):
    # Example: Create a DataFrame or pass the text directly
    # Adjust this based on how your model accepts input
    data = pd.DataFrame({'content': [text]})
    results = model.predict(data)  # Replace with your model's evaluation method
    return results

# Streamlit UI
def main():
    st.title("PDF Model Evaluation App")
    st.write("Upload a PDF file, and the model will evaluate it.")

    # File upload
    uploaded_file = st.file_uploader("Upload your file", type=['pdf'])

    if uploaded_file is not None:
        st.write("File uploaded successfully. Extracting text...")

        try:
            # Extract text from the PDF
            text = extract_text_from_pdf(uploaded_file)
            
            if not text.strip():
                st.error("No text could be extracted from the PDF. Please upload a valid file.")
                return
            
            st.write("Text extraction successful!")
            st.write("### Extracted Text Preview")
            st.write(text[:1000])  # Show the first 500 characters

            # Load the model
            model = load_model()

            # Evaluate the extracted text
            results = evaluate_text(model, text)
            
            # Display results
            st.write("### Evaluation Results")
            st.write("#### Predictions")
            st.write(results['prediction'])
            st.write("#### Reasons")
            st.write(results['reasons'])
            st.write("#### Similarity")
            st.write(f"Similarity Score: {results['similarity']}")
            st.write("#### Innovation")
            st.write(f"Innovation Score: {results['innovation_score']}")
            st.write("#### Novelty")
            st.write(f"Novelty Score: {results['novelty_score']}")
            st.write("#### Plagiarism")
            st.write(f"Plagiarism Score: {results['plagiarism_score']}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()
