from pypdf import PdfReader
from langchain_core.documents import Document

def extract_text_from_pdfs(uploaded_files) -> list[Document]:
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                # Skip empty/unreadable pages safely
                if text and text.strip():
                    doc = Document(
                        page_content=text.strip(),
                        metadata={
                            "source_doc": uploaded_file.name,
                            "page_number": page_num
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"Error reading {uploaded_file.name}: {e}")
            
    return documents