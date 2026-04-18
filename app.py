"""Streamlit UI for the standalone class notes RAG app."""

from agent import build_agent
from evaluator import score_groundedness
import os
import streamlit as st
from retriever import get_hybrid_retriever, get_notes_by_date, get_available_dates, reconstruct_notes
from chain import build_rag_chain, extract_sources, format_docs
from config import get_notes_root, get_source_specs

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Class Notes RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 50%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .sub-header {
        color: #94A3B8;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* Card styling for results */
    .source-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s ease;
    }

    .source-card:hover {
        border-color: rgba(124, 58, 237, 0.7);
    }

    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }

    .source-title {
        color: #E2E8F0;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 4px;
    }

    .source-preview {
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 6px;
        line-height: 1.4;
    }

    /* Mode selector styling */
    .mode-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Animated gradient divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, #7C3AED, #4F46E5, #2563EB, #4F46E5, #7C3AED);
        background-size: 200% auto;
        animation: gradient-shift 3s linear infinite;
        border-radius: 2px;
        margin: 1rem 0;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    /* Example chip buttons */
    .example-chip {
        display: inline-block;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #C4B5FD;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 4px;
    }

    .example-chip:hover {
        background: rgba(124, 58, 237, 0.3);
        border-color: rgba(124, 58, 237, 0.7);
    }

    /* Notes rendering */
    .notes-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        padding: 24px;
        margin-top: 12px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }

    /* Metric cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


api_key = st.session_state.api_key


def queue_question(question: str) -> None:
    cleaned = question.strip()
    if not cleaned:
        return
    st.session_state.pending_question = cleaned


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Sources ({len(sources)} documents)"):
        for src in sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <span class="source-badge">{src['collection']}</span>
                    <div class="source-title">{src['title']}</div>
                    <div class="source-preview">{src['chunk_preview']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_groundedness(groundedness: dict) -> None:
    score = groundedness.get("score", 0.5)
    # Conditions must go from most restrictive to least — check low scores first
    if score < 0.60:
        emoji = "🔴"
        label = "Hallucination Risk"
    elif score < 0.85:
        emoji = "🟡"
        label = "Partially Grounded"
    else:
        emoji = "🟢"
        label = "Grounded"

    st.markdown(
        f'<p style="font-size:0.9rem; color:#94A3B8; margin-top:8px;">'
        f'{emoji} <strong>{label}</strong> — Groundedness: {score:.2f}</p>',
        unsafe_allow_html=True,
    )
    with st.expander("ℹ️ Rationale"):
        st.markdown(groundedness.get("rationale", "No rationale available."))

def get_kb_stats() -> tuple[int, int]:
    """Return document and collection counts from the current source registry."""
    try:
        specs = get_source_specs()
    except Exception:
        return 0, 0

    existing_specs = [spec for spec in specs if spec["path"].exists()]
    document_count = len(existing_specs)
    collection_count = len({spec["collection"] for spec in existing_specs})
    return document_count, collection_count

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    notes_root = get_notes_root()

    st.markdown('<p class="main-header" style="font-size:1.5rem;">🧠 Class Notes RAG</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="font-size:0.85rem;">Standalone notes search and study assistant</p>', unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Mode selection
    st.markdown('<p class="mode-label">Mode</p>', unsafe_allow_html=True)
    mode = st.radio(
        "Select mode",
        ["💬 Ask a Question", "📅 Browse Notes by Date", "🤖 Agentic Mode"],
        label_visibility="collapsed",
        key="mode_selector",
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Stats
    st.markdown('<p class="mode-label">Knowledge Base</p>', unsafe_allow_html=True)
    document_count, collection_count = get_kb_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{document_count}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{collection_count}</div>
            <div class="stat-label">Collections</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div class="stat-card">
        <div class="stat-value" style="font-size:1rem;">Hybrid</div>
        <div class="stat-label">BM25 + Semantic</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="mode-label">Notes Root</p>', unsafe_allow_html=True)
    st.code(str(notes_root), language=None)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About this app"):
        st.markdown("""
        Built with **LangChain**, **ChromaDB**, and **GPT-4o-mini**.

        **Retrieval**: Hybrid search combining BM25 keyword matching
        with semantic vector search (OpenAI embeddings) for best results.

        **Chunking**: Recursive character splitting with markdown-aware
        separators (1000 chars, 150 overlap).

        **Model**: `gpt-4o-mini` for generation,
        `text-embedding-3-small` for embeddings.

        Set `CLASS_NOTES_DIR` to index notes outside this repo.
        """)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<h1 class="main-header">Class Notes RAG</h1>', unsafe_allow_html=True)

_MODE_SUBTITLES = {
    "💬 Ask a Question": "Direct RAG — hybrid keyword + semantic search over your indexed notes",
    "📅 Browse Notes by Date": "Browse full notes by session date when dated notes are available",
    "🤖 Agentic Mode": "Agentic RAG — ReAct loop with tool routing and multiple retrieval calls",
}
st.markdown(
    f'<p class="sub-header">{_MODE_SUBTITLES.get(mode, "")}</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)



# ---------- Validate API key ----------
if not api_key or api_key == "your-openai-api-key-here":
    st.error(
        "🔑 Set `OPENAI_API_KEY` in your environment before using this app. "
        "Optional fallback: create `.streamlit/secrets.toml` from `.streamlit/secrets.example.toml`."
    )
    st.stop()


# =====================================================================
# MODE 1: Ask a Question
# =====================================================================

if mode == "💬 Ask a Question":

    # Example questions
    st.markdown("**Try asking:**")

    example_questions = [
        "How are embeddings used to retrieve relevant documents?",
        "What is cosine similarity and why is it used?",
        "How does hybrid retrieval improve a RAG system?",
        "How does the ReAct loop work in agentic RAG?",
        "What evaluation metrics are used for RAG?",
        "What is the difference between fine-tuning and RAG?",
    ]

    # Render as clickable buttons in columns
    cols = st.columns(3)
    for i, q in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                queue_question(q)
                st.rerun()

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🧠"):
            st.markdown(message["content"])
            if "sources" in message:
                render_sources(message["sources"])
            if "groundedness" in message:
                render_groundedness(message["groundedness"])

    # Chat input
    if question := st.chat_input("Ask about your indexed notes..."):
        queue_question(question)
        st.rerun()

    if st.session_state.pending_question:
        pending_question = st.session_state.pending_question
        st.session_state.pending_question = None
        st.session_state.messages.append({"role": "user", "content": pending_question})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(pending_question)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔍 Searching notes..."):
                try:
                    retriever = get_hybrid_retriever(api_key)
                    chain = build_rag_chain(retriever, api_key)

                    retrieved_docs = retriever.invoke(pending_question)
                    sources = extract_sources(retrieved_docs)
                    answer = chain.invoke(pending_question)
                    context = format_docs(retrieved_docs)
                    groundedness = score_groundedness(pending_question, context, answer, api_key)

                    st.markdown(answer)
                    render_sources(sources)
                    render_groundedness(groundedness)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "groundedness": groundedness
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

elif mode == "🤖 Agentic Mode":
    st.info("🤖 Agentic mode uses tool routing and may make multiple retrieval calls to answer one question.")

    for message in st.session_state.agent_messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            if "steps" in message:
                with st.expander("🔍 Agent reasoning steps"):
                    st.text(message["steps"])

    if question := st.chat_input("Ask the agent...", key="agent_input"):
        st.session_state.agent_messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤖 Agent is reasoning..."):
                try:
                    agent = build_agent(api_key)
                    result = agent.invoke(
                    {"messages": [("user", question)]},
                        config={"recursion_limit": 20},  # ~5 ReAct iterations
                    )
                    # LangGraph returns {"messages": [...]}: last message is the final answer
                    messages = result["messages"]
                    answer = messages[-1].content
                    # Intermediate messages are tool calls and observations
                    steps = "\n\n".join(
                        f"{m.__class__.__name__}: {m.content}"
                        for m in messages[1:-1]
                    )
                    st.markdown(answer)
                    with st.expander("🔍 Agent reasoning steps"):
                        st.text(steps)
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "steps": steps,
                    })
                except Exception as e:
                    st.error(f"❌ Agent error: {str(e)}")


    # Clear chat button
    if st.session_state.agent_messages:
        if st.button("🗑️ Clear agent conversation", use_container_width=True):
            st.session_state.agent_messages = []
            st.rerun()

# =====================================================================
# MODE 2: Browse Notes by Date
# =====================================================================

elif mode == "📅 Browse Notes by Date":

    st.markdown("### 📅 Browse Full Notes")
    st.markdown("Select a date to view the complete notes from that session.")

    try:
        available_dates = get_available_dates(api_key)

        if not available_dates:
            st.warning("No dated notes found. Run `python ingest.py` first or add dates to your note files.")
            st.stop()

        # Format dates for display
        date_display = {d: f"📅 {d}" for d in available_dates}

        selected_date = st.selectbox(
            "Select note date",
            options=available_dates,
            format_func=lambda x: date_display[x],
            key="date_selector",
        )

        if selected_date:
            with st.spinner("📖 Loading notes..."):
                docs = get_notes_by_date(api_key, selected_date)

                if not docs:
                    st.warning(f"No notes found for {selected_date}.")
                else:
                    reconstructed = reconstruct_notes(docs)

                    # Display stats
                    st.markdown(f"""
                    <div style="display:flex; gap:12px; margin-bottom:16px;">
                        <div class="stat-card" style="flex:1;">
                            <div class="stat-value">{len(reconstructed)}</div>
                            <div class="stat-label">Documents</div>
                        </div>
                        <div class="stat-card" style="flex:1;">
                            <div class="stat-value">{len(docs)}</div>
                            <div class="stat-label">Chunks</div>
                        </div>
                        <div class="stat-card" style="flex:1;">
                            <div class="stat-value">{sum(len(v) for v in reconstructed.values()):,}</div>
                            <div class="stat-label">Characters</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

                    # Render each document
                    for title, content in reconstructed.items():
                        with st.expander(f"📄 {title}", expanded=True):
                            st.markdown(content)

    except Exception as e:
        st.error(f"❌ Error loading notes: {str(e)}")
        st.info("Make sure you've run `python ingest.py` to build the index first.")
