import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("Semantic Research Paper Assistant")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + " "
    
    chunks = chunk_text(extracted_text)
    st.success(f"Extracted {len(chunks)} chunks from the document.")
    
    with st.spinner("Generating dense embeddings..."):
        chunk_embeddings = model.encode(chunks)
    
    query = st.text_input("Ask a question about the paper:")
    if query:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:3]
        
        st.subheader("Top Relevant Contexts:")
        for idx in top_indices:
            st.markdown(f"**Relevance Score: {similarities[idx]:.4f}**")
            st.info(chunks[idx])
