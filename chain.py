"""LCEL chain builder for the standalone class notes RAG app."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable teaching assistant.
Your job is to answer questions using ONLY the provided notes below.

Rules:
1. Answer based strictly on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say:
   "I don't have enough information in the indexed notes to answer that. Try rephrasing your question or check a different note."
3. When relevant, mention which collection or note the information comes from.
4. Use clear formatting with bullet points, tables, or code blocks when helpful.
5. Keep answers concise but thorough — aim for the level of a study guide.
6. If a question asks for a definition, give the definition first, then add context.

Context from indexed notes:
{context}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string with source markers."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("title", "Unknown")
        collection = doc.metadata.get("collection", "Notes")
        header = f"[Source {i}: {source} ({collection})]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def extract_sources(docs: list[Document]) -> list[dict]:
    """Extract source information for display in the UI."""
    sources = []
    seen = set()
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        collection = doc.metadata.get("collection", "Notes")
        source_file = doc.metadata.get("source_file", "")
        key = f"{title}|{source_file}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": title,
                "collection": collection,
                "source_file": source_file,
                "chunk_preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200 else doc.page_content,
            })
    return sources


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------


def build_rag_chain(retriever, api_key: str):
    """
    Build the full RAG chain using LCEL.

    Args:
        retriever: A LangChain retriever (e.g. EnsembleRetriever).
        api_key: OpenAI API key.

    Returns:
        A tuple of (chain, retriever) where:
          - chain: the full RAG chain that returns an answer string
          - retriever: the retriever for extracting source docs separately
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return chain
