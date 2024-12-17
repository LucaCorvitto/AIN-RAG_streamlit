import os
import time
from getpass import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

os.environ['PINECONE_API_KEY'] = "pcsk_288X2e_M4zbjKrEAnunFUJcRYJ5twDDCTfhD1ybW8gsUSkKuZkfpPbGF3dn4WeQgjfAwx8"

def populate_database(list_of_paths):
    
    ### Initialize OpenAI Embeddings ###
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")
    model_name = 'text-embedding-3-small'  # You can choose a different embedding model
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    ### Connect to Pinecone ###
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    index_name = "streamlit-rag-us" # Choose a unique index name
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Create index if it doesn't exist
    if index_name not in existing_indexes:
        pc.create_index(index_name, dimension=1536, metric='cosine', spec=spec)
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)


    index = pc.Index(index_name)
    index.describe_index_stats()

    ### Load and process multiple PDFs ###
    #file_paths = list_of_paths #['file1.pdf', 'file2.pdf', 'file3.pdf']  # List your PDF file paths here
    all_splits = []  # Store all document chunks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file_path in list_of_paths:
        # Load each PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Add metadata to each document (e.g., the file name)
        for doc in docs:
            doc.metadata = {"source": file_path}  # Attach metadata for each document
        
        # Split the documents into chunks
        splits = text_splitter.split_documents(docs)
        
        # Collect all splits
        all_splits.extend(splits)

    ### Store document embeddings in Pinecone ###
    vectorstore = PineconeVectorStore(index=index, embedding=embed)
    vectorstore.add_documents(all_splits)  # Add document chunks to Pinecone vector store

    return