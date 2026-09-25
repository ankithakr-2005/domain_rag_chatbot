import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt import QA_PROMPT

load_dotenv()

def generate_answer(query: str, vector_store, top_k: int = 4):
    """
    Retrieves relevant document chunks and generates a clean text answer using Gemini LLM.
    """
    # Section 5: Retrieval
    docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    if not docs_and_scores:
        return "I could not find this information in the uploaded documents.", []

    retrieved_docs = [doc for doc, score in docs_and_scores]

    # Combine contexts
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Section 6: Answer Generation
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is missing from environment variables.", []

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        google_api_key=api_key,
        temperature=0.0
    )

    formatted_prompt = QA_PROMPT.format(context=context_text, question=query)
    response = llm.invoke(formatted_prompt)
    
    # --- Clean String Extraction Fix ---
    raw_content = response.content

    if isinstance(raw_content, str):
        clean_text = raw_content
    elif isinstance(raw_content, list):
        # Extract text field from list of content blocks
        extracted = []
        for item in raw_content:
            if isinstance(item, dict) and "text" in item:
                extracted.append(item["text"])
            elif isinstance(item, str):
                extracted.append(item)
        clean_text = "\n".join(extracted) if extracted else str(raw_content)
    else:
        clean_text = str(raw_content)

    # Prepare metadata sources for display
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "source_doc": doc.metadata.get("source_doc", "Unknown"),
            "page_number": doc.metadata.get("page_number", "N/A"),
            "snippet": doc.page_content[:150] + "..."
        })

    return clean_text, sources