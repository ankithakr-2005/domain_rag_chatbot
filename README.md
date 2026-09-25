# Domain-Specific RAG Chatbot for PDF Question Answering

A reliable, grounded question-answering assistant that allows users to upload PDF documents (course notes, company policies, manuals, legal documents, or resumes) and ask questions in natural language. The system retrieves relevant passages and generates answers strictly based on the uploaded document content with source citations.

---

## Key Features
- **Document Ingestion:** Multi-PDF support with automated page-by-page text extraction (`pypdf`).
- **Metadata Tracking:** Preserves original source file names and page numbers for accurate citations.
- **Text Chunking:** Context-aware text splitting using `RecursiveCharacterTextSplitter`.
- **Vector Search:** Pretrained embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` and fast similarity retrieval using `FAISS`.
- **Grounded Answer Generation:** Strict prompt guardrails using Gemini (`gemini-3.6-flash`) to eliminate hallucinations.
- **Fallback Refusal Mechanism:** Automatically returns *"I could not find this information in the uploaded documents."* when query context is unavailable.
- **Interactive UI:** Built with Streamlit featuring sidebar document management, expandable source citations, and chat history controls.

---

## Setup & Installation

1. Create and Activate Virtual Environment:
    python -m venv venv
    venv\Scripts\activate

2. Install Dependencies:
   pip install -r requirements.txt

3. Configure Environment Variables
   Create a .env file in the root directory and add your Gemini API key:
   GEMINI_API_KEY=your_gemini_api_key_here

## Running the Application
### Launch the Streamlit web app:
   streamlit run app.py

