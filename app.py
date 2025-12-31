import streamlit as st

from src.data_processor import generate_chunks_from_pdfs
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME
from src.metrics.generate_eval_dataset import generate_dataset
from src.metrics.evaluate_rag import run_evaluation

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
    page_title=" Mobility RAG Assistant",
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
        llm_model=LLM_MODEL_NAME,
        top_k=4,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title(" Mobility Assistant")
st.sidebar.markdown(
    """
This assistant answers questions about **mobility and transport**
using **only official documents**.

All answers are grounded in the document database and include citations.
"""
)

if st.sidebar.button("Rebuild document index"):
    with st.spinner("Processing documents and rebuilding vector store..."):
        chunks = generate_chunks_from_pdfs()
        chunk_dicts = chunks_to_dicts(chunks)
        st.session_state.vector_store.add_chunks(chunk_dicts)

    st.sidebar.success(f"Indexed {len(chunks)} document chunks.")

if st.sidebar.button("Generate Dataset"):
    with st.spinner("Generating evaluation dataset with questions, answers and citations"):
        generate_dataset()  # Or direct call to script logic
        
    st.sidebar.success("eval_dataset.json generated successfully!")

if st.sidebar.button("Run Evaluation"):
    with st.spinner("Running RAG evaluation (correctness, citations, consistency)..."):
        results = run_evaluation()  # Returns dict with table + variance
        st.subheader("Evaluation Results")
        st.table(results["summary_table"])
        st.info(results["message"])
    st.sidebar.success("Evaluation complete!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Models**")
st.sidebar.markdown("- Embeddings: " + EMBEDDING_MODEL_NAME)
st.sidebar.markdown("- LLM: " + LLM_MODEL_NAME)


# -----------------------------
# Main UI
# -----------------------------
st.title(" Mobility Knowledge Assistant")

st.markdown(
    """
Ask a question about **transport, mobility, or infrastructure**.
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
user_input = st.chat_input("Ask a question about mobility in ...")

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
