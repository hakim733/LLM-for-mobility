import streamlit as st

from src.data_processor import generate_chunks_from_pdfs
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline


# -----------------------------
# Helpers
# -----------------------------
def chunks_to_dicts(chunks):
    """
    Convert DocumentChunk objects to dicts expected by VectorStore.
    """
    out = []
    for c in chunks:
        out.append(
            {
                "id": c.chunk_id,
                "text": c.text,
                "metadata": {
                    "source": c.source,
                    "page": c.page,
                },
            }
        )
    return out


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Skåne Mobility RAG Assistant",
    layout="wide",
)


# -----------------------------
# Initialize shared objects
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(
        st.session_state.vector_store,
        llm_model="llama3",
        top_k=4,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Skåne Mobility Assistant")
st.sidebar.markdown(
    """
This assistant answers questions about **mobility and transport in Skåne**
using **only official regional documents**.

All answers are grounded in the document database and include citations.
"""
)

if st.sidebar.button("Rebuild document index"):
    with st.spinner("Processing documents and rebuilding vector store..."):
        chunks = generate_chunks_from_pdfs()
        chunk_dicts = chunks_to_dicts(chunks)
        st.session_state.vector_store.add_chunks(chunk_dicts)

    st.sidebar.success(f"Indexed {len(chunks)} document chunks.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Models**")
st.sidebar.markdown("- Embeddings: all-MiniLM-L6-v2")
st.sidebar.markdown("- LLM: llama3 (local via Ollama)")


# -----------------------------
# Main UI
# -----------------------------
st.title("🚍 Skåne Mobility Knowledge Assistant")

st.markdown(
    """
Ask a question about **transport, mobility, or infrastructure in Skåne**.
The assistant will answer **only if the information is present in the documents**.
"""
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            with st.expander("Sources"):
                for s in msg.get("sources", []):
                    st.markdown(f"- **{s['source']}**, page {s['page']}")


# Chat input
user_input = st.chat_input("Ask a question about mobility in Skåne...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = st.session_state.rag.answer(user_input)
            st.markdown(result["answer"])

            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source']}**, page {s['page']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
