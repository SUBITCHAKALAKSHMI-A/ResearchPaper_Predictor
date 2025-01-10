import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load the model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to extract abstract or key section
def extract_abstract(text):
    # Attempt to find an "Abstract" section
    abstract_pattern = re.compile(r"(?i)(abstract)\s*[:\n](.+?)(\n\n|\Z)", re.DOTALL)
    match = abstract_pattern.search(text)
    if match:
        return match.group(2).strip()
    
    # Fallback: Use the first 3 sentences as a pseudo-abstract
    sentences = re.split(r'[.!?]', text)
    return " ".join(sentences[:3]).strip()

# Function to evaluate the extracted text using the model
def evaluate_text(model, text):
    # Extract abstract or key section
    abstract = extract_abstract(text)

    # Encode the input abstract
    text_embedding = model.encode([abstract], convert_to_tensor=True)

    # Simulated comparison dataset (replace with actual research paper embeddings)
    example_papers = [
        "This paper explores advanced AI techniques for language models.",
        "A comprehensive review of machine learning algorithms.",
        "An in-depth study on neural networks and their applications.",
    ]
    example_embeddings = model.encode(example_papers, convert_to_tensor=True)

    # Calculate similarities between the uploaded paper and the examples
    similarities = cosine_similarity(text_embedding.cpu().numpy(), example_embeddings.cpu().numpy())
    avg_similarity = np.mean(similarities)

    # Derive evaluation scores
    plagiarism_score = avg_similarity
    innovation_score = max(0, 1 - avg_similarity)
    novelty_score = max(0, 1 - avg_similarity)

    # Generate acceptance or rejection decision
    is_accepted = innovation_score > 0.5 and plagiarism_score < 0.3

    # Generate dynamic reasons based on content
    reasons = []
    if is_accepted:
        reasons.append("The research paper has been accepted due to the following reasons:")
        reasons.append(f"- The abstract discusses key concepts such as: {abstract[:200]}...")
        reasons.append("- The research presents innovative ideas with minimal overlap to existing studies.")
        reasons.append(f"- The innovation score is high ({innovation_score:.2f}), and plagiarism is low ({plagiarism_score:.2f}).")
    else:
        reasons.append("The research paper has been rejected due to the following reasons:")
        reasons.append(f"- The abstract suggests a significant overlap with existing works: {abstract[:200]}...")
        reasons.append(f"- The plagiarism score ({plagiarism_score:.2f}) exceeds the acceptable threshold.")
        if innovation_score <= 0.5:
            reasons.append(f"- The innovation score ({innovation_score:.2f}) is insufficient.")

    return {
        'prediction': "Accepted" if is_accepted else "Rejected",
        'reasons': reasons,
        'similarity': avg_similarity,
        'innovation_score': innovation_score,
        'novelty_score': novelty_score,
        'plagiarism_score': plagiarism_score,
        'abstract': abstract
    }

# Streamlit UI
def main():
    st.title("Research Paper Evaluation App")
    st.write("Upload a PDF file, and the system will evaluate it.")

    uploaded_file = st.file_uploader("Upload your file", type=['pdf'])

    if uploaded_file is not None:
        st.write("File uploaded successfully. Extracting text...")

        try:
            text = extract_text_from_pdf(uploaded_file)
            
            if not text.strip():
                st.error("No text could be extracted from the PDF. Please upload a valid file.")
                return
            
            st.write("Text extraction successful!")
            st.write("### Extracted Text Preview")
            st.write(text[:1000])  # Show the first 1000 characters

            model = load_model()

            # Evaluate the extracted text
            results = evaluate_text(model, text)
            
            # Display results
            st.write("### Evaluation Results")
            st.write(f"#### Prediction: {results['prediction']}")
            st.write("#### Reasons:")
            for reason in results['reasons']:
                st.write(f"- {reason}")
            st.write(f"#### Extracted Abstract:")
            st.write(results['abstract'])
            st.write(f"#### Similarity Score: {results['similarity']:.2f}")
            st.write(f"#### Innovation Score: {results['innovation_score']:.2f}")
            st.write(f"#### Novelty Score: {results['novelty_score']:.2f}")
            st.write(f"#### Plagiarism Score: {results['plagiarism_score']:.2f}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()
