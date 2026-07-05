import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline
from src.evaluation import (
    TestQuestion,
    generate_test_questions,
    run_validation,
    compute_retrieval_metrics,
)
from src.metrics_report import collect_system_metrics

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG Document Question Answering")
st.caption("Upload your own PDFs or notes, then ask questions grounded in that content.")

# ---- Session state ----
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.indexed_files = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

EMBEDDING_MODEL_OPTIONS = {
    "MiniLM-L6-v2 (fast, 384-dim, default)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet-base-v2 (slower, 768-dim, more accurate)": "sentence-transformers/all-mpnet-base-v2",
    "BGE-small-en-v1.5 (384-dim, strong retrieval)": "BAAI/bge-small-en-v1.5",
}

GENERATION_MODEL_OPTIONS = {
    "Flan-T5-small (fastest, weakest)": "google/flan-t5-small",
    "Flan-T5-base (default, balanced)": "google/flan-t5-base",
    "Flan-T5-large (slower, better instruction-following)": "google/flan-t5-large",
}

with st.sidebar:
    st.header("Settings")

    st.subheader("Models")
    generation_backend = st.radio(
        "Generation backend",
        ["Local (Flan-T5)", "Cohere API (command-r)"],
        index=0,
        help="Cohere's command-r gives much longer, better-structured answers "
        "and summaries, but needs an API key and internet access.",
    )
    embedding_choice = st.selectbox("Embedding model", list(EMBEDDING_MODEL_OPTIONS.keys()), index=0)

    cohere_api_key = None
    if generation_backend == "Cohere API (command-r)":
        generation_choice = "Cohere command-r-08-2024"
        env_key = os.environ.get("COHERE_API_KEY")
        if env_key:
            st.caption("✅ Using COHERE_API_KEY from .env / environment")
            cohere_api_key = env_key
        else:
            cohere_api_key = st.text_input(
                "Cohere API key", type="password",
                help="Not found in .env — paste it here, or add COHERE_API_KEY to a .env file instead.",
            )
    else:
        generation_choice = st.selectbox("Chat / generation model", list(GENERATION_MODEL_OPTIONS.keys()), index=1)
    st.caption("Larger models give better answers but take longer to download (first run) and run.")

    st.subheader("Chunking & Retrieval")
    chunk_size = st.slider("Chunk size (characters)", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 200, 75, step=25)
    mode = st.radio("Retrieval mode", ["vector", "hybrid"], index=0)
    rerank = st.checkbox("Enable keyword-overlap re-ranking", value=False)
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 4)
    offline_mode = st.checkbox(
        "Offline test mode (no model downloads)",
        value=False,
        help="Uses dependency-free fallback embedder/generator for testing "
        "the pipeline without internet access. Answer quality will be poor — "
        "leave unchecked for real use.",
    )

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

uploaded_files = st.file_uploader(
    "Upload PDF or .txt files", type=["pdf", "txt", "md"], accept_multiple_files=True
)

if uploaded_files:
    current_names = tuple(sorted(f.name for f in uploaded_files))
    current_config = (
        current_names, embedding_choice, generation_choice, chunk_size, chunk_overlap,
        offline_mode, generation_backend,
    )
    needs_reindex = (
        st.session_state.pipeline is None or st.session_state.get("indexed_config") != current_config
    )

    if needs_reindex:
        if generation_backend == "Cohere API (command-r)" and not offline_mode and not cohere_api_key:
            st.error("Enter a Cohere API key in the sidebar (or set COHERE_API_KEY in a .env file) to continue.")
            st.stop()

        with st.spinner("Indexing documents... (first run downloads models, may take a minute)"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for f in uploaded_files:
                    with open(os.path.join(tmp_dir, f.name), "wb") as out:
                        out.write(f.getbuffer())

                pipeline = RAGPipeline(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    offline_mode=offline_mode,
                    embedding_model_name=EMBEDDING_MODEL_OPTIONS[embedding_choice],
                    generation_model_name=(
                        GENERATION_MODEL_OPTIONS[generation_choice]
                        if generation_backend != "Cohere API (command-r)"
                        else "command-r-08-2024"
                    ),
                    generation_backend="cohere" if generation_backend == "Cohere API (command-r)" else "local",
                    cohere_api_key=cohere_api_key,
                )
                pipeline.ingest_folder(tmp_dir)

            st.session_state.pipeline = pipeline
            st.session_state.indexed_files = current_names
            st.session_state.indexed_config = current_config
            st.session_state.chat_history = []  # new documents/models -> fresh conversation

    st.success(f"Indexed {len(uploaded_files)} file(s): {', '.join(current_names)}")

    with st.expander("📊 Validation log & system metrics report"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Run Validation"):
                with st.spinner("Running validation..."):

                    test_qs = generate_test_questions(st.session_state.pipeline)

                    results = run_validation(
                        pipeline=st.session_state.pipeline,
                        test_questions=test_qs,
                        top_k=top_k,
                        mode=mode,
                        rerank=rerank,
                    )

                    metrics = compute_retrieval_metrics(results)

                    st.success("Validation completed!")
                    st.json(metrics)

        with col2:
            if st.button("Generate system metrics report"):
                metrics = collect_system_metrics(st.session_state.pipeline)
                st.json(metrics)
                import json
                st.download_button(
                    "Download metrics report (.json)",
                    json.dumps(metrics, indent=2, default=str),
                    file_name="system_metrics_report.json",
                )

    st.divider()

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander("Show retrieved context (evidence)"):
                for c in turn["context"]:
                    st.markdown(f"**{c['source']} — chunk {c['chunk_id']}** (score: {c.get('score', 0):.3f})")
                    st.text(c["text"])

    question = st.chat_input("Ask a question about your documents")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                result = st.session_state.pipeline.ask(
                    question, top_k=top_k, mode=mode, rerank=rerank, return_context=True
                )
            st.write(result["answer"])
            with st.expander("Show retrieved context (evidence)"):
                for c in result["context"]:
                    st.markdown(f"**{c['source']} — chunk {c['chunk_id']}** (score: {c.get('score', 0):.3f})")
                    st.text(c["text"])

        st.session_state.chat_history.append(
            {"question": question, "answer": result["answer"], "context": result["context"]}
        )
else:
    st.info("Upload one or more PDF/.txt files to get started.")