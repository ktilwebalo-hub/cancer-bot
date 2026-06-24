from collections import Counter
import glob
import os
import re
from pathlib import Path
import json
import sys

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

llm = ChatOpenAI(temperature=0.3, model_name=MODEL)

KNOWLEDGE_BASE = Path(__file__).parent.parent / "knowledge-base"
TOP_K = 5


def _load_cache():
    """Load cached knowledge base from JSON file."""
    # Get the directory where answer.py is located
    current_dir = Path(__file__).parent
    cache_file = current_dir / "knowledge_base_cache.json"
    
    print(f"Looking for cache file at: {cache_file}", file=sys.stderr)
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded {len(data)} chunks from cache", file=sys.stderr)
            documents = []
            for item in data:
                doc = Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"Error loading cache: {e}", file=sys.stderr)
            return []
    else:
        print(f"Cache file not found at: {cache_file}", file=sys.stderr)
        # Try alternative location: root directory
        alt_cache = Path(__file__).parent.parent / "knowledge_base_cache.json"
        if alt_cache.exists():
            print(f"Found cache at alternative location: {alt_cache}", file=sys.stderr)
            with open(alt_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents = []
            for item in data:
                doc = Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
                documents.append(doc)
            return documents
        return []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_document(query_tokens: Counter[str], document: Document) -> tuple[int, int]:
    doc_tokens = Counter(_tokenize(document.page_content))
    shared_terms = sum(min(query_tokens[token], doc_tokens[token]) for token in query_tokens)
    return shared_terms, len(document.page_content)


# Load documents from cache
DOCUMENTS = _load_cache()
print(f"Total documents loaded: {len(DOCUMENTS)}", file=sys.stderr)

SYSTEM_PROMPT = """
You are a knowledgeable, supportive, and compassionate cancer awareness assistant. Your goal is to help people understand cancer prevention, screening, treatment options, and support resources.

Key principles:
- Be empathetic and supportive
- Provide evidence-based information
- Emphasize the importance of early detection and screening
- Always recommend consulting healthcare professionals
- Give practical, actionable advice
- Be sensitive to the emotional aspects of cancer

Context:
{context}

When providing advice, always:
1. Acknowledge the person's question with empathy
2. Provide clear, actionable information
3. Include important safety considerations
4. Offer resources for additional support
5. Be supportive and encouraging
"""


def resolve_query(query: str, history: list):
    """Resolve query with conversation history."""
    last_user = None
    if history:
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user = turn["content"]
                break
    if last_user:
        return f"{last_user}\n{query}"
    return query


def fetch_context(query: str) -> list[Document]:
    """Retrieve relevant documents from local cache."""
    if not DOCUMENTS:
        print("Warning: No documents loaded from cache. Run ingest.py first.", file=sys.stderr)
        return []
    
    query_tokens = Counter(_tokenize(query))
    scored_docs = sorted(
        DOCUMENTS,
        key=lambda document: _score_document(query_tokens, document),
        reverse=True,
    )
    
    # Debug: print top documents
    print(f"Retrieved {len(scored_docs[:TOP_K])} documents for query", file=sys.stderr)
    for i, doc in enumerate(scored_docs[:TOP_K]):
        print(f"  Doc {i+1}: {doc.metadata.get('filename', 'Unknown')} - {len(doc.page_content)} chars", file=sys.stderr)
    
    return scored_docs[:TOP_K]


def answer_question(query: str, history=None):
    """Generate answer using retrieved context and conversation history."""
    resolved_query = resolve_query(query, history or [])
    docs = fetch_context(resolved_query)
    
    # Debug: print context
    print(f"Found {len(docs)} documents for answer", file=sys.stderr)
    
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Print first 200 chars of context for debugging
    if context:
        print(f"Context preview: {context[:200]}...", file=sys.stderr)
    else:
        print("WARNING: No context found!", file=sys.stderr)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    
    # Return both the answer and the documents
    return response.content, docs