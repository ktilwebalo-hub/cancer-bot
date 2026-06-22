from collections import Counter
import glob
import os
import re
from pathlib import Path
import json

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
    cache_file = Path(__file__).parent / "knowledge_base_cache.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
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
        print("Warning: No documents loaded from cache. Run ingest.py first.")
        return []
    
    query_tokens = Counter(_tokenize(query))
    scored_docs = sorted(
        DOCUMENTS,
        key=lambda document: _score_document(query_tokens, document),
        reverse=True,
    )
    return scored_docs[:TOP_K]


def answer_question(query: str, history=None):
    """Generate answer using retrieved context and conversation history."""
    resolved_query = resolve_query(query, history or [])
    docs = fetch_context(resolved_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    return response.content, docs