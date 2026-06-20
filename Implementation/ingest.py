import os
import glob
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Reuse one index (already created in your Pinecone project)
index_name = "cancerllm-index"

# Create if missing
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Now connect
index = pc.Index(index_name)


# Namespace to separate datasets
NAMESPACE = "canceraware"


def fetch_documents():
    """Load pdf files from knowledge base folders."""
    documents = []
    pdf_files = glob.glob(os.path.join(KNOWLEDGE_BASE, "*.pdf"))

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_path)
                doc.metadata["doc_type"] = "pdf"
                doc.metadata["folder"] = KNOWLEDGE_BASE
                documents.extend(docs)
        except Exception as e:
            print(f"Error loading {os.path.baseman(pdf_path)}: {e}")

    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return splitter.split_documents(documents)


def embed_and_prepare(chunks):
    vectors = []
    for i, doc in enumerate(chunks):
        vec = embeddings.embed_query(doc.page_content)
        vectors.append({
            "id": f"doc-{i}",
            "values": vec,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content,
                "doc_type": doc.metadata.get("doc_type", "")
            }
        })
    return vectors


def insert_parallel(vectors, batch_size=100):
    async def run_upserts():
        with ThreadPoolExecutor(max_workers=5) as executor:
            loop = asyncio.get_running_loop()
            tasks = []
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                tasks.append(loop.run_in_executor(executor, index.upsert, batch, NAMESPACE))
            await asyncio.gather(*tasks)

    asyncio.run(run_upserts())



if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    vectors = embed_and_prepare(chunks)
    insert_parallel(vectors)
    print(f"🚀 Ingestion complete: {len(vectors)} vectors inserted into Pinecone (namespace={NAMESPACE})")

