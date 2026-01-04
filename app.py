import streamlit as st
import pandas as pd
import altair as alt
import os

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
    with st.spinner("Running full RAG evaluation (correctness, citations, consistency, variance)..."):
        results = run_evaluation()
        
        st.subheader(" RAG Evaluation Results")
        
        # Per-temperature summary (variance across temps)
        st.markdown("### Per-Temperature Metrics")
        if results["temp_summary"]:
            temp_df = pd.DataFrame(results["temp_summary"])
            if 'Temperature' in temp_df.columns:
                st.table(temp_df.set_index('Temperature'))
            else:
                st.table(temp_df)
        else:
            st.warning("No temp summary data!")
        
        csv_path = results.get("csv_path", "")
        df = pd.read_csv(csv_path)

        # Row 2: CONSOLIDATED (mean across bases) + extras
        col3, col4 = st.columns(2)
        with col3:
            # Consolidated Correctness (mean per temp)
            df_mean = df.groupby('temp')[['correctness', 'consistency']].mean().reset_index()
            corr_consol = alt.Chart(df_mean).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('temp:Q', title="Temperature"),
                y=alt.Y('correctness:Q', title="Mean Correctness", scale=alt.Scale(domain=[0,1])),
                tooltip=['temp', 'correctness']
            ).properties(
                title="Correctness vs Temp",
                width=400,
                height=250
            )
            st.altair_chart(corr_consol, use_container_width=True)
        
        with col4:
            # Consolidated Consistency (mean per temp)
            cons_consol = alt.Chart(df_mean).mark_line(point=True, strokeWidth=3, color="orange").encode(
                x=alt.X('temp:Q', title="Temperature"),
                y=alt.Y('consistency:Q', title="Mean Consistency", scale=alt.Scale(domain=[0,1])),
                tooltip=['temp', 'consistency']
            ).properties(
                title="Consistency vs Temp",
                width=400,
                height=250
            )
            st.altair_chart(cons_consol, use_container_width=True)

        # Full variance across temperatures
        st.markdown("### Variance Across Temperatures")
        vars = results['variances']
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Correctness Var", f"{vars['correctness_var']:.3f}")
        with col2: st.metric("Citation Var", f"{vars['citation_var']:.3f}")
        with col3: st.metric("Consistency Var", f"{vars['consistency_var']:.3f}")
        #st.info(results["message"])
        
        # CSV download (safe)
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                csv_data = f.read()
            st.download_button(
                label="📥 Download Full Metrics CSV",
                data=csv_data,
                file_name="rag_eval_metrics.csv",
                mime="text/csv"
            )
        else:
            st.warning("CSV not generated - check eval logs.")
        
        # Quick charts (with fallback)
        try:
            # Per-base query summary (paraphrase robustness)
            st.markdown("### Per-Base Query Metrics")
            if results["base_summary"]:
                base_df = pd.DataFrame(results["base_summary"])
                if 'Base Query' in base_df.columns:
                    st.table(base_df.set_index('Base Query'))
                else:
                    st.table(base_df)  # Fallback
            else:
                st.warning("No base summary data - run dataset generation first!")
        
            col1, col2 = st.columns(2)
            with col1:
                corr_chart = alt.Chart(df).mark_line().encode(
                    x="temp:Q", y="correctness:Q", color="base_query:N"
                ).properties(title="Correctness vs Temperature")
                st.altair_chart(corr_chart, use_container_width=True)
            with col2:
                cons_chart = alt.Chart(df).mark_line().encode(
                    x="temp:Q", y="consistency:Q", color="base_query:N"
                ).properties(title="Consistency vs Temperature")
                st.altair_chart(cons_chart, use_container_width=True)
        except:
            st.info("Charts unavailable - download CSV for analysis.")
    
    st.sidebar.success("Full evaluation complete!")
    st.balloons()

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
