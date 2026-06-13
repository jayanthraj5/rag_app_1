import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

st.set_page_config(page_title="Enterprise Policy Assistant")

st.title("📄 Enterprise Policy & SOP Assistant")
st.write("Upload a PDF and ask questions about its contents.")

pdf = st.file_uploader("Upload PDF", type="pdf")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

if pdf:

    # Read PDF
    reader = PdfReader(pdf)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Sentence-based chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) < 500:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence

    if chunk:
        chunks.append(chunk.strip())

    st.success(f"PDF Loaded Successfully ({len(chunks)} chunks)")

    model = load_model()

    embeddings = model.encode(chunks).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    query = st.text_input("Ask a Question")

    if query:

        q_embedding = model.encode([query]).astype("float32")

        faiss.normalize_L2(q_embedding)

        scores, indices = index.search(q_embedding, k=min(3, len(chunks)))

        st.subheader("📌 Most Relevant Information")

        for rank, idx in enumerate(indices[0], start=1):

            st.markdown(f"### Result {rank}")
            st.write(chunks[idx])

            st.caption(
                f"Similarity Score: {scores[0][rank-1]:.3f}"
            )

            st.divider()
