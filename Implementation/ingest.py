import os
import glob
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv(override=True)

KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")
OUTPUT_FILE = Path(__file__).with_name("knowledge_base_cache.json")


def fetch_documents():
    """Load all markdown files from the knowledge base."""
    documents = []
    for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def embed_and_prepare(chunks):
    """Generate a lightweight cache for local retrieval."""
    return [
        {
            "id": f"chunk-{i}",
            "text": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
                "filename": os.path.basename(doc.metadata.get("source", "")),
            },
        }
        for i, doc in enumerate(chunks)
    ]


def write_cache(vectors):
    """Persist a cache of chunks for local startup."""
    OUTPUT_FILE.write_text(json.dumps(vectors, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    print("📚 Loading documents from knowledge base...")
    docs = fetch_documents()
    print(f"✅ Loaded {len(docs)} documents")

    print("✂️ Creating chunks...")
    chunks = create_chunks(docs)
    print(f"✅ Created {len(chunks)} chunks")

    print("🧠 Preparing cache...")
    vectors = embed_and_prepare(chunks)
    print(f"✅ Prepared {len(vectors)} cached chunks")

    print(f"💾 Writing cache to {OUTPUT_FILE.name}...")
    write_cache(vectors)
    print(f"🚀 Ingestion completed: {len(vectors)} cached chunks written")