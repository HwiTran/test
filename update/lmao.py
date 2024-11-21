import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "llama3.2"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"
UPLOAD_DIRECTORY = "./pdfs"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


def ingest_pdf(file_path):
    """Load PDF documents."""
    try:
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Failed to load PDF: {str(e)}")
        st.error("Failed to load the PDF file. Please try again.")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def load_vector_db():
    """Load or create the vector database."""
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Created new vector database.")
    return vector_db


def update_vector_db(vector_db):
    """Update the vector database with all PDFs in the upload directory."""
    coll = vector_db.get(
    )  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])

    ids_to_del = coll['ids']
    logging.info(ids_to_del)

    if ids_to_del:
        vector_db._collection.delete(ids_to_del)
        logging.info("Deleted existing documents from the vector database.")
    else:
        logging.info("No documents to delete from the vector database.")

    for pdf_file in os.listdir(UPLOAD_DIRECTORY):
        pdf_path = os.path.join(UPLOAD_DIRECTORY, pdf_file)
        data = ingest_pdf(pdf_path)
        if data is None:
            continue
        chunks = split_documents(data)
        vector_db.add_documents(documents=chunks)
        vector_db.persist()
        logging.info(f"Updated vector database with {pdf_file}.")


def list_uploaded_pdfs():
    """List all uploaded PDF files."""
    return os.listdir(UPLOAD_DIRECTORY)


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. 
        First, translate the prompt from vietnammese into english. 
        Then generate five different versions of the same translated prompt. 
        By generating multiple perspectives on the user question, your goal is to 
        help the user overcome some of the limitations of the distance-based 
        similarity search. Provide these alternative questions separated by newlines. 
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever



def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    template = """Hãy trả lời câu hỏi của người dùng bằng tiếng việt một cách thân thiện và dễ hiểu dựa trên những thông tin sau:
{context}
Câu hỏi: {question}
Nếu người dùng hỏi về thông tin không tồn tại trong văn bản, hãy trả lời rằng không tìm thấy thông tin.
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("Document Assistant")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Display list of uploaded PDFs with delete button
    st.sidebar.title("Uploaded PDFs")
    pdf_files = list_uploaded_pdfs()
    if pdf_files:
        vector_db = load_vector_db()
        for pdf_file in pdf_files:
            file_path = os.path.join(UPLOAD_DIRECTORY, pdf_file)
            col1, col2 = st.sidebar.columns([8, 1])
            col1.write(pdf_file)
            if col2.button("x", key=pdf_file):
                os.remove(file_path)
                st.sidebar.success(
                    f"File {pdf_file} deleted successfully. Please reload page to see changes")
                # Update vector database after deletion
                update_vector_db(vector_db)
    else:
        st.sidebar.write("No PDFs uploaded yet.")

    # User input
    user_input = st.text_input("Enter your question:", "", placeholder="Prompt")

    if uploaded_file:
        pdf_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"File {uploaded_file.name} uploaded successfully. Please reload page to see changes")

        # Load vector database and update with new PDF
        vector_db = load_vector_db()
        update_vector_db(vector_db)

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                llm = ChatOllama(model=MODEL_NAME)

                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return
  
                retriever = create_retriever(vector_db, llm)  # Create the retriever
 
                chain = create_chain(retriever, llm)          # Create the chain

                response = chain.invoke(input=user_input)     # Get the response

                st.markdown("## **Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
