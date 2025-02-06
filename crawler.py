import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# List of specific pages to crawl
pages_to_crawl = [
    "http://myaastha.in/",
    "http://myaastha.in/about-us/",
    "http://myaastha.in/how-to-become-a-member/",
    "http://myaastha.in/scheme/savings-account/",
    "http://myaastha.in/scheme/recurring-deposits/",
    "http://myaastha.in/scheme/fixed-deposit/",
    "http://myaastha.in/scheme/monthly-income/",
    "http://myaastha.in/personal-loan/",
    "http://myaastha.in/advance-against-deposits/",
    "http://myaastha.in/loan-against-property/",
    "http://myaastha.in/contact-us/"
]

# Function to extract text from specific tags (span, p, table)
def extract_text_from_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from specific tags
        text_elements = []
        for tag in soup.find_all(['span', 'p', 'table']):
            if tag.name == 'table':
                # Extract table content
                rows = tag.find_all('tr')
                table_text = []
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    row_text = [cell.get_text(strip=True) for cell in cells]
                    table_text.append(" | ".join(row_text))
                text_elements.append("\n".join(table_text))
            else:
                # Extract text from span and p tags
                text_elements.append(tag.get_text(strip=True))

        # Combine all text elements into a single string
        page_text = "\n".join(text_elements)
        return page_text
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

# Function to crawl specific pages and extract text
def crawl_specific_pages(pages):
    documents = []
    for url in pages:
        print(f"Crawling: {url}")
        page_text = extract_text_from_page(url)
        if page_text:
            documents.append(page_text)
    return documents

# Crawl the specific pages
documents = crawl_specific_pages(pages_to_crawl)

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents(documents)

# Initialize Hugging Face embeddings and vector store
try:
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    print("Data has been successfully crawled and stored in Chroma DB.")
except Exception as e:
    print(f"Error initializing embeddings or vector store: {e}")