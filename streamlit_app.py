import streamlit as st
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from google import genai
from google.genai import types
from together import Together


# ---------------------
# Environment Setup
# ---------------------
google_api_key = st.secrets['google_api_key']
qdrant_api_key = st.secrets['qdrant_api_key']
together_api_key = st.secrets['together_api_key']

# Qdrant Configuration
QDRANT_URL = st.secrets['QDRANT_URL']
COLLECTION_NAME = "RAG_Skincare"
EMBEDDING_MODEL = "text-embedding-004"

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=qdrant_api_key,
    prefer_grpc=True,
    timeout=120.0
)

# ---------------------
# Utilities
# ---------------------
@st.cache_resource(show_spinner=False)
def get_genai_client():
    return genai.Client(api_key=google_api_key)

@st.cache_resource(show_spinner=False)
def get_together_client():
    return Together(api_key=together_api_key)

def embed_text(text):
    try:
        client = get_genai_client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                outputDimensionality=1536,
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        return response.embeddings[0].values
    except Exception as e:
        st.error(f"⚠️ Gagal membuat embedding: {e}")
        return None

def retrieve_documents(query):
    embedding = embed_text(query)
    if not embedding:
        return []
    search_params = models.SearchParams(hnsw_ef=512, exact=True)
    global limit_k
    limit_k = 8
    try:
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            query_filter=None,
            limit=limit_k,
            score_threshold=0.7,
            search_params=search_params,
            with_payload=True,
            with_vectors=True
        )
        return [data.payload for data in results]
    except Exception as e:
        st.error(f"❌ Gagal mengambil dokumen: {e}")
        return []

def format_context(docs):
    return "\n".join([
        f"- {doc.get('product_name', 'Tanpa Nama')}, {doc.get('description', '')} (Gambar: {doc.get('product_image_link', '-')})"
        for doc in docs
    ])

def generate_answer(query, docs):
    if not docs:
        yield "Maaf, saya tidak menemukan informasi relevan."
        return

    prompt = f"""Gunakan informasi di bawah ini untuk menjawab pertanyaan pengguna seakurat mungkin. selalu sertakan product_image_link agar user dapat mengakses gambar product:

{docs}

Pertanyaan: {query}

Jawaban yang akurat dan alami:"""

    client = get_together_client()
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
        yield content

# ---------------------
# Exit State Handler
# ---------------------
if "exited" in st.session_state and st.session_state["exited"]:
    st.markdown("## 🙏 Terima Kasih Telah Menggunakan Chatbot RAG Produk Skincare")
    st.stop()

# ---------------------
# Page Config
# ---------------------
st.set_page_config(page_title="💊 Chatbot RAG Produk Skincare", layout="wide")
st.title("💬 Chatbot RAG Produk Skincare")
st.markdown("Masukkan pertanyaan seputar **produk skincare** dan sistem akan memberikan jawaban berdasarkan pengetahuan yang relevan.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # will store dicts: {"role": "user"/"assistant", "content": text}

# ---------------------
# Conversation Loop
# ---------------------
for i in range(len(st.session_state.messages) // 2 + 1):
    is_last_input = (i == len(st.session_state.messages) // 2)

    # Show input box for this turn
    with st.form(f"chat_form_{i}", clear_on_submit=True):
        user_input = st.text_input(
            "📝 Pertanyaan:",
            key=f"input_{i}",
            placeholder="Tulis pertanyaan Anda di sini...",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Kirim")

    # Autofocus only on the last input box
    if is_last_input:
        st.markdown(f"""
            <script>
            setTimeout(function(){{
                var input = window.parent.document.querySelector('input[id="input_{i}"]');
                if (input) {{
                    input.focus();
                }}
            }}, 100);
            </script>
        """, unsafe_allow_html=True)

    if submitted and user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate and save assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari jawaban..."):
                try:
                    docs = retrieve_documents(user_input)
                    response_text = ""
                    for chunk in generate_answer(user_input, docs):
                        response_text += chunk
                except:
                    response_text = "Maaf, saya tidak menemukan informasi relevan."
            # Add unique ID for scrolling
            st.markdown(f'<div id="last-message">{response_text}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()  # Refresh so new Q&A is rendered immediately

    # After input, show Q&A history for this turn
    idx = i * 2
    if idx < len(st.session_state.messages):
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[idx]["content"])
    if idx + 1 < len(st.session_state.messages):
        with st.chat_message("assistant"):
            # Add ID only to the very last message for scrolling
            if i == len(st.session_state.messages) // 2 - 1:
                st.markdown(f'<div id="last-message">{st.session_state.messages[idx + 1]["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(st.session_state.messages[idx + 1]["content"])

# ---------------------
# Exit Button
# ---------------------
if st.button("❌ Exit"):
    st.session_state.clear()
    st.session_state["exited"] = True
    st.rerun()

# ---------------------
# Auto-Scroll to Last Message
# ---------------------
st.markdown("""
<script>
setTimeout(function() {
    var last = window.parent.document.getElementById("last-message");
    if (last) {
        last.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
}, 100);
</script>
""", unsafe_allow_html=True)

# Footer
st.caption("🧠 Creator : Hafiz Pambudi")
