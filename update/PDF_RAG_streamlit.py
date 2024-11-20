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
EMBEDDING_MODEL = "llama3.2"#"nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
# def load_vector_db(pdf_file):
#     """Load or create the vector database."""
#     # Pull the embedding model if not already available
#     ollama.pull(EMBEDDING_MODEL)

#     embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

#     if os.path.exists(PERSIST_DIRECTORY):
#         vector_db = Chroma(
#             embedding_function=embedding,
#             collection_name=VECTOR_STORE_NAME,
#             persist_directory=PERSIST_DIRECTORY,
#         )
#         logging.info("Loaded existing vector database.")
#     else:
#         # Load and process the uploaded PDF document
#         data = ingest_pdf(pdf_file)
#         if data is None:
#             return None

#         # Split the documents into chunks
#         chunks = split_documents(data)

#         vector_db = Chroma.from_documents(
#             documents=chunks,
#             embedding=embedding,
#             collection_name=VECTOR_STORE_NAME,
#             persist_directory=PERSIST_DIRECTORY,
#         )
#         vector_db.persist()
#         logging.info("Vector database created and persisted.")
#     return vector_db

def load_vector_db(pdf_file):
    """Load or update the vector database."""
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Kiểm tra nếu thư mục tồn tại
    if os.path.exists(PERSIST_DIRECTORY):
        # Tải cơ sở dữ liệu hiện tại
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")

        # Kiểm tra nếu tệp PDF đã được xử lý
        data = ingest_pdf(pdf_file)
        if data is None:
            return None
        
        chunks = split_documents(data)
        
        # Thêm tài liệu mới vào cơ sở dữ liệu
        vector_db.add_documents(documents=chunks)
        vector_db.persist()  # Lưu cập nhật
        logging.info("Vector database updated with new documents.")
    else:
        # Nếu không có cơ sở dữ liệu, tạo cơ sở dữ liệu mới
        data = ingest_pdf(pdf_file)
        if data is None:
            return None
        
        chunks = split_documents(data)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("New vector database created and persisted.")
    return vector_db

def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
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
    # RAG prompt
    template = """Hãy trả lời câu hỏi của người dùng dựa trên những thông tin sau:
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


    # Tải lên PDF và nhúng tài liệu
    if uploaded_file:
        with st.spinner("Processing and embedding the PDF..."):
            try:
                # Lưu tệp PDF tạm thời
                pdf_path = f"./uploaded_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Nhúng tài liệu PDF
                vector_db = load_vector_db(pdf_path)
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                else:
                    st.success("PDF has been successfully embedded!")
            except Exception as e:
                st.error(f"An error occurred while embedding the PDF: {str(e)}")

    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""

    # Nhập câu hỏi
    user_input = st.text_input("Enter your question:", "")

    if user_input and user_input != st.session_state.last_user_input:
        # Cập nhật trạng thái user_input
        st.session_state.last_user_input = user_input

        # Thực hiện xử lý
        with st.spinner("Generating response..."):
            try:
                if not os.path.exists(PERSIST_DIRECTORY):
                    st.error("No embedded documents found. Please upload and embed a PDF first.")
                    return

                # Tải cơ sở dữ liệu vector
                vector_db = Chroma(
                    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
                    collection_name=VECTOR_STORE_NAME,
                    persist_directory=PERSIST_DIRECTORY,
                )

                # Tạo retriever và chain
                llm = ChatOllama(model=MODEL_NAME)
                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)

                # Lấy phản hồi
                response = chain.invoke(input=user_input)
                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # if uploaded_file and user_input:
    #     with st.spinner("Generating response..."):
    #         try:
    #             # Save uploaded file temporarily
    #             pdf_path = f"./uploaded_{uploaded_file.name}"
    #             with open(pdf_path, "wb") as f:
    #                 f.write(uploaded_file.read())

    #             # Initialize the language model
    #             llm = ChatOllama(model=MODEL_NAME)

    #             # Load the vector database
    #             vector_db = load_vector_db(pdf_path)
    #             if vector_db is None:
    #                 st.error("Failed to load or create the vector database.")
    #                 return

    #             # Create the retriever
    #             retriever = create_retriever(vector_db, llm)

    #             # Create the chain
    #             chain = create_chain(retriever, llm)

    #             # Get the response
    #             response = chain.invoke(input=user_input)

    #             st.markdown("**Assistant:**")
    #             st.write(response)
    #         except Exception as e:
    #             st.error(f"An error occurred: {str(e)}")
    # elif not uploaded_file:
    #     st.info("Please upload a PDF file to get started.")
    # elif not user_input:
    #     st.info("Please enter a question to get started.")

if __name__ == "__main__":
    main()
    
