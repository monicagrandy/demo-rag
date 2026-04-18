import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from retriever import get_hybrid_retriever, get_notes_by_date, get_available_dates, reconstruct_notes
from chain import format_docs

def make_tools(api_key: str) -> list:
    @tool
    def search_notes(query: str) -> str:
        """
        Search the indexed notes using hybrid semantic search.
        Use for concept definitions, comparisons, how-to questions, and topic
        explanations (e.g., "What is MMR?", "How does chunking work?").
        Do NOT use this when the user references a specific note date — use
        get_notes_by_date_tool instead.
        Input: a natural language query or keyword string describing the topic.
        Output: formatted excerpts from indexed notes labeled by source
        and collection. If the result is incomplete, retry with a more specific query.
        """
        docs = get_hybrid_retriever(api_key).invoke(query)
        return format_docs(docs) if docs else "No results found."
        
    @tool  
    def get_notes_by_date_tool(date: str) -> str:
        """
        Retrieves all indexed notes for the provided date using a metadata filter.
        Use when asked about content from a specific date.
        Do NOT use this for general concept questions — use search_notes instead.
        Input: a stringified date (YYYY-MM-DD)
        Output: Complete and formatted indexed notes from the input date. If the date isn't found,
        call list_note_dates first to see available dates.
        """
        docs = get_notes_by_date(api_key, date)
        notes = reconstruct_notes(docs)
        return "\n\n".join(f"## {title}\n{content}" for title, content in notes.items()) if notes else f"No notes found for {date}."
        
    @tool
    def list_note_dates(dummy: str = "") -> str:
        """
        List all the dates associated with sources.
        Use when asked about what's been coverered or which sessions exist.
        Do NOT use this to get note content — use search_notes or
        get_notes_by_date_tool for that. This only returns dates.
        Input: ignore — pass an empty string or omit.
        Output: String of all dates associated with all sources.
        """
        dates = get_available_dates(api_key)
        return "Available note dates: " + ", ".join(dates) if dates else "No dated notes indexed yet."
        
    return [search_notes, get_notes_by_date_tool, list_note_dates]

def build_agent(api_key: str):
    """
    Build a LangGraph ReAct agent backed by the note-search tools.
    Returns a compiled graph — invoke with {"messages": [("user", question)]}.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    tools = make_tools(api_key)
    # LangGraph's create_react_agent replaces the old langchain AgentExecutor pattern.
    # max_iterations is controlled via recursion_limit on the config dict at invoke time.
    return create_react_agent(llm, tools)
