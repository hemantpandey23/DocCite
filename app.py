"""
Cited Doc Q&A Bot
-----------------
What this app does, in one paragraph:
You upload a PDF. The app splits it into chunks, converts each chunk into a
vector (an "embedding") that captures its meaning, and stores those vectors in
memory. When you ask a question, the question is also turned into a vector, and
the app finds the 3 chunks whose vectors are closest to it. Those chunks are
sent to an LLM along with your question, with strict instructions to answer
only from them. This pattern is called RAG (Retrieval-Augmented Generation).

Run:  streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# CONFIG: the knobs you will tune when testing. Change one at a time.
# ----------------------------------------------------------------------------
CHUNK_SIZE = 500        # words per chunk. Smaller = more precise, less context.
CHUNK_OVERLAP = 50      # words shared between neighbouring chunks, so a
                        # sentence cut at a boundary still appears whole in one.
TOP_K = 3               # how many chunks are sent to the LLM.
MIN_SIMILARITY = 0.25   # if even the best chunk scores below this, we refuse
                        # without calling the LLM. Tune this on your test set.
MODEL = "gemini-3.8-flash"  # swap for any model your API key can access.


# ----------------------------------------------------------------------------
# STEP 1: Load the embedding model (once).
# @st.cache_resource keeps the model in memory across reruns. Streamlit reruns
# this whole script on every interaction, so without caching it would reload
# the model every time you typed a question.
# ----------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    # all-MiniLM-L6-v2 is small (~80 MB), free, and runs on CPU.
    return SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------------------------------------------------------
# STEP 2: Read the PDF and split it into overlapping chunks.
# ----------------------------------------------------------------------------
def extract_text(file) -> str:
    reader = PdfReader(file)
    # extract_text() returns None for pages with no text layer (e.g. scans),
    # so "or ''" prevents a crash. If everything is empty, the PDF is likely
    # scanned and this app cannot read it (that would need OCR).
    return " ".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap  # each chunk starts `step` words after the last one
    chunks = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size])
        if piece:
            chunks.append(piece)
    return chunks


# ----------------------------------------------------------------------------
# STEP 3: Embed all chunks (once per uploaded document).
# @st.cache_data stores the result keyed by the file's bytes. Asking 20
# questions about the same PDF embeds it only once, not 20 times.
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading and indexing document...")
def build_index(file_bytes: bytes):
    import io
    text = extract_text(io.BytesIO(file_bytes))
    chunks = chunk_text(text)
    if not chunks:
        return [], np.empty((0, 0))
    model = load_embedder()
    # normalize_embeddings=True makes every vector length 1. Then the dot
    # product of two vectors equals their cosine similarity (1 = same meaning,
    # 0 = unrelated), which makes retrieval a single matrix multiply.
    vectors = model.encode(chunks, normalize_embeddings=True)
    return chunks, vectors


# ----------------------------------------------------------------------------
# STEP 4: Retrieve the chunks most relevant to the question.
# ----------------------------------------------------------------------------
def retrieve(question: str, chunks, vectors, k: int = TOP_K):
    model = load_embedder()
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q_vec                     # similarity to every chunk
    best = np.argsort(scores)[::-1][:k]          # indices of the k highest
    return [(int(i), float(scores[i]), chunks[i]) for i in best]


# ----------------------------------------------------------------------------
# STEP 5: Ask the LLM, with the retrieved chunks as the only allowed source.
# ----------------------------------------------------------------------------
def get_api_key():
    # Reads the Gemini key from Streamlit secrets (or an env variable locally).
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def ask_llm(question: str, hits) -> str:
    # Label each chunk so the model can cite it as [Chunk 12].
    context = "\n\n".join(f"[Chunk {i}]\n{text}" for i, _, text in hits)

    # Same grounding prompt as before: use only the context, refuse if the
    # answer is missing, and quote the supporting sentence.
    prompt = (
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I can't find that in the document.\"\n"
        "Otherwise, give a short answer, then quote the supporting sentence "
        "and cite its chunk like [Chunk 3].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=prompt)
    # response.text can be empty if the model's output was blocked.
    return response.text or "No answer was returned. Try rephrasing."

    genai.configure(api_key=get_api_key())
    model = genai.GenerativeModel(model_name=MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=500
        )
    )
    return response.content[0].text


# ----------------------------------------------------------------------------
# STEP 6: The UI. Streamlit runs this top to bottom on every interaction.
# ----------------------------------------------------------------------------
st.title("Cited Doc Q&A")
st.caption("Upload a text-based PDF and ask questions. Answers come only from the document.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about the document")

if uploaded and question:
    chunks, vectors = build_index(uploaded.getvalue())

    if not chunks:
        st.error("No text found. This PDF may be a scan; try a text-based PDF.")
    else:
        hits = retrieve(question, chunks, vectors)

        # Cheap refusal: if even the best match is weak, the document almost
        # certainly doesn't contain the answer. Skipping the LLM call saves
        # money and avoids a confident-sounding wrong answer.
        if hits[0][1] < MIN_SIMILARITY:
            st.warning("I can't find that in the document.")
        else:
            with st.spinner("Thinking..."):
                st.markdown(ask_llm(question, hits))

        # Always show the evidence so the user can verify the answer.
        st.subheader("Sources retrieved")
        for idx, score, text in hits:
            with st.expander(f"Chunk {idx} (similarity {score:.2f})"):
                st.write(text)