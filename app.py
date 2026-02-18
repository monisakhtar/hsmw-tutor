# app.py
import streamlit as st
import gin
from config import create_client
from ingestion import ingest_pdf
from chain import build_graph

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="HSMW Tutor",
    page_icon="🎓",
    layout="wide",
)

# ── Load Gin Config ───────────────────────────────────────
if "gin_loaded" not in st.session_state:
    gin.enter_interactive_mode()
    gin.parse_config_file("sicim/config.gin")
    st.session_state["gin_loaded"] = True

# ── Initialize Client ─────────────────────────────────────
if "client" not in st.session_state:
    st.session_state["client"] = create_client()

client = st.session_state["client"]
host = client._client.base_url.__str__()
api_key = client._client.headers["Authorization"].replace("Bearer ", "")

# ── Initialize Graph ──────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state["graph"] = None

# ── Initialize Chat History ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── UI ────────────────────────────────────────────────────
st.title("🎓 HSMW Tutor")
st.caption("Upload your lecture PDFs and ask questions!")

# Sidebar — PDF Upload
with st.sidebar:
    st.header("📄 Upload Lecture PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("📥 Ingest PDF"):
            with st.spinner("Processing PDF..."):
                client = st.session_state["client"]
                ingest_pdf(uploaded_file, client=client)
                st.session_state["graph"] = build_graph(client=client)
                st.success("✅ PDF ingested! You can now ask questions.")
    st.divider()

    st.subheader("🗄️ Database")
    if st.button("🗑️ Clear ChromaDB", type="secondary"):
        import shutil
        import os
        from config import CHROMA_PERSIST_DIR

        # Reset graph first to release ChromaDB client
        st.session_state["graph"] = None
        st.session_state["messages"] = []

        # Small delay to let ChromaDB release file locks
        import time
        time.sleep(0.5)

        # Now safely delete and recreate
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
        os.makedirs(CHROMA_PERSIST_DIR)

        st.success("✅ Database cleared!")
        st.rerun()

# Main — Chat Interface
if st.session_state["graph"] is None:
    st.info("👈 Please upload and ingest a PDF to get started.")
else:
    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about your lecture..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state["graph"].invoke({
                    "question": question,
                    "rewritten_question": "",
                    "documents": [],
                    "answer": "",
                    "grade": "",
                    "retries": 0,
                })
                answer = result["answer"]
                st.markdown(answer)

        # Add assistant message
        st.session_state["messages"].append({"role": "assistant", "content": answer})
