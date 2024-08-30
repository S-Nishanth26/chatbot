import camelot
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve Pinecone API key from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
index_name = "pdf-table"

# Retrieve index
index = pc.Index(index_name)

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract tables from a PDF using camelot
def extract_tables_from_pdf(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all')
    table_texts = []
    
    for table in tables:
        df = table.df
        table_text = df.to_string(index=False, header=False)
        table_texts.append(table_text)
    
    return table_texts

# Function to store tables in Pinecone
def store_tables_in_pinecone(table_texts, doc_id):
    for i, table_text in enumerate(table_texts):
        vector = embedding_model.encode(table_text).tolist()
        index.upsert([
            {
                'id': f"{doc_id}_table_{i}",
                'values': vector,
                'metadata': {'text': table_text}
            }
        ], namespace="tables")

# Function to extract and store tables from a PDF
def process_pdf(pdf_path, doc_id):
    table_texts = extract_tables_from_pdf(pdf_path)
    store_tables_in_pinecone(table_texts, doc_id)

# Example usage
pdf_path = r"D:\Downloads\anatomy_vol_1.pdf"
doc_id = "document_1"
process_pdf(pdf_path, doc_id)

print(f"Tables from {pdf_path} have been stored in Pinecone index {index_name}.")
