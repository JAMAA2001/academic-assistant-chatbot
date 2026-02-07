import streamlit as st
import re
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq
from datetime import datetime
import torch
import os  
import time

# ===============================
# Charger les données et le modèle
# ===============================
with open("all_chunks_text.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

index = faiss.read_index("academic_faiss1.index")
embedder =SentenceTransformer("BAAI/bge-base-en-v1.5",device="cpu")
api_key = os.getenv("GROQ_API_KEY")  # <-- lire la clé depuis l'environnement
if not api_key:
    st.error(
        "❌ GROQ_API_KEY n'est pas définie.\n"
        "👉 Ajoutez-la dans les variables d'environnement ou dans les secrets Streamlit."
    )
    st.stop()
# ===============================
# Fonctions RAG
# ===============================
def semantic_search_simple(query, embedder, index, all_chunks, top_k=10, score_threshold=0.3):
    query_embedding = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    results = [
        {"text": all_chunks[i], "index": int(i), "score": float(scores[0][rank])}
        for rank, i in enumerate(indices[0])
        if i != -1 and scores[0][rank] >= score_threshold
    ]
    return results

def bm25_search_simple(query, all_chunks, top_k=10):
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())
    tokenized_docs = [tokenize(doc) for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": all_chunks[i], "index": int(i), "score": float(scores[i])}
        for i in top_indices if scores[i] > 0
    ]

def rrk_simple(bm25_results, semantic_results, top_k=10, k=50, beta=0.4):
    scores_dict = {}
    for rank, r in enumerate(bm25_results):
        scores_dict[r["index"]] = scores_dict.get(r["index"], 0) + beta / (k + rank + 1)
    for rank, r in enumerate(semantic_results):
        scores_dict[r["index"]] = scores_dict.get(r["index"], 0) + (1 - beta) / (k + rank + 1)
    sorted_indices = sorted(scores_dict, key=scores_dict.get, reverse=True)[:top_k]
    merged_texts = []
    for i in sorted_indices:
        for r in bm25_results + semantic_results:
            if r["index"] == i:
                merged_texts.append(r["text"])
                break
    return merged_texts, sorted_indices

def test_rrk(query, top_k=5):
    bm25_results = bm25_search_simple(query, all_chunks, top_k)
    semantic_results = semantic_search_simple(query, embedder, index, all_chunks, top_k)
    merged_texts, _ = rrk_simple(bm25_results, semantic_results, top_k)
    return merged_texts

def chunks_to_bullet_prompt(query, chunks):
    bullets = "\n".join([f"- {c}" for c in chunks])
    return f"""Réponds à la question suivante :
{query}

Utilise uniquement les informations suivantes :
{bullets}
"""

def generate_rag_response(query, api_key):
    chunks = test_rrk(query)
    prompt = chunks_to_bullet_prompt(query, chunks)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant universitaire. Réponds en français uniquement.\n"
                    "Règles:\n"
                    "- Ne dis pas 'il semble'. Sois direct.\n"
                    "- Réponds avec des listes claires quand c'est une liste de modules."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

# ===============================
# Interface Streamlit (ChatGPT style)
# ===============================
st.set_page_config(
    page_title="Assistant académique intelligent",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Style (نفس CSS ديالك)
st.markdown("""
<style>
.title {
    background: linear-gradient(to right, #4facfe, #00f2fe);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.card {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.2);
}
.stButton>button {
    background: #4facfe;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎓 Assistant académique intelligent</div>', unsafe_allow_html=True)
now = datetime.now()
st.markdown(
    f"<p style='text-align:center; color: gray;'>Date et heure : {now.strftime('%d/%m/%Y %H:%M:%S')}</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### 📖 Description")
    st.write("Assistant universitaire intelligent basé sur RAG hybride (BM25 + embeddings FAISS)")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Salut 👋 Pose-moi ta question."}]
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Salut 👋 Pose-moi ta question."}]

# Display messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_query = st.chat_input("Écris ta question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ ...")
        try:
            start_time = time.time()

            answer = generate_rag_response(user_query, api_key)

            elapsed_time = time.time() - start_time
            now_answer = datetime.now()

            placeholder.markdown(answer)
            st.caption(
                f"Réponse générée le : {now_answer.strftime('%d/%m/%Y à %H:%M:%S')} "
                f"- Temps écoulé : {elapsed_time:.2f} secondes"
            )

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            err = f"❌ Une erreur est survenue : {e}"
            placeholder.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
