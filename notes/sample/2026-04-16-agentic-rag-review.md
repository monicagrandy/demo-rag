### Assignment Review Overview

- Assignment completion status mixed across class
- Focus on RAG pipeline implementation using 20 tickets as database
- Embedding ticket ID, title, category, priority, description, resolution
- Using text-embedding-3-small model and GPT-4 mini with temperature 0

### Retrieval Strategies & Performance

- Cosine similarity retriever with top 3 documents default
- Format documents function combines retrieved docs with newlines
- Three prompt variations tested:
  1. Basic concise answers with ticket IDs
  2. Step-by-step explanations showing relevance
  3. Bullet point responses with sources
- MMR (Maximum Marginal Relevance) for diversity when documents have duplicated content
  - Lambda parameter controls similarity vs diversity balance (default 0.5)
  - Fetch 10 documents, narrow to 3 for balanced results

### Claude 4.7 Performance Issues

- Major regression in context handling: 91.9% → 59.2% at 256K context
- Even worse at 1M context: 78.3% → 32.2%
- 4x more token consumption than previous version
- Worse performance in agentic search: 83.7% → 17%
- More emotionally responsive, agrees more frequently
- Aligned with Chinese government positions on Taiwan

### Fallback Systems & Confidence Scoring

- Minimum threshold approach (0.7 similarity score)
- High confidence responses when documents exceed threshold
- Medium confidence warnings when relevance uncertain
- Works best with crowdsourced databases vs internally maintained catalogs

### Document Processing Strategies

- Stuff strategy: Include all documents in single prompt (most common)
- Map-reduce: Summarize each document separately, then combine summaries
- Refine: Generate draft answer, iteratively improve with additional documents
  - Has recency bias - last document influences final answer most
- Stuff strategy recommended for most use cases due to simplicity and cost

### Metadata Filtering & Streaming

- Filter by category (authentication, database, payment)
- Filter by priority levels
- Streaming responses improve user experience
  - Show tokens as generated vs waiting for complete response
  - Requires streaming=true in LLM config plus callback handler

### Multi-turn Conversations

- Condensed chain converts chat history + current question → standalone question
- Standalone question goes to retriever
- Conversation chain generates final answer using context + chat history
- Hallucination detection compares source text with generated answer

### Evaluation Metrics & Production Considerations

- Precision: Relevant documents / Total retrieved documents
- Recall: Retrieved relevant documents / Total relevant documents available
- F1 score: Harmonic mean of precision and recall
- Average Precision: Accounts for ranking order, not just retrieval
- Latency tracking and cost monitoring essential for production
- P95 latency more important than average latency
- OpenAI volume discounts available with pre-commitment (20% discount example for $100K/month)