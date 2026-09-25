import streamlit as st
from document_loader import extract_text_from_pdfs
from vector_store import build_vector_store
from rag_pipeline import generate_answer

st.set_page_config(page_title="Domain-Specific RAG Chatbot", layout="wide")

st.title("📚 Domain-Specific Document Assistant")
st.caption("Answers are strictly grounded in uploaded PDF document content.")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Controls
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = extract_text_from_pdfs(uploaded_files)
                if docs:
                    st.session_state.vector_store = build_vector_store(docs)
                    st.success(f"Indexed {len(uploaded_files)} PDF(s) successfully!")
                else:
                    st.error("Could not extract readable text from uploaded PDFs.")
        else:
            st.warning("Please upload at least one PDF file.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Render Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📍 View Document Sources"):
                for idx, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**Source {idx}:** {src['source_doc']} | **Page:** {src['page_number']}")
                    st.caption(f"*\"{src['snippet']}\"*")

# Chat Input Processing
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            ans = "Please upload and process at least one PDF document first."
            st.markdown(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
        else:
            with st.spinner("Searching documents..."):
                answer, sources = generate_answer(user_query, st.session_state.vector_store)
                
                # Render clean string
                st.markdown(answer)
                
                if sources:
                    with st.expander("📍 View Document Sources"):
                        for idx, src in enumerate(sources, start=1):
                            st.markdown(f"**Source {idx}:** {src['source_doc']} | **Page:** {src['page_number']}")
                            st.caption(f"*\"{src['snippet']}\"*")

                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })