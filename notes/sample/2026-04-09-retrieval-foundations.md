# Retrieval Foundations

Thu, 09 Apr 26

## Embeddings

- Embeddings convert text into vectors so semantically similar inputs land near each other in vector space.
- At query time, the system embeds the user question with the same model used for the notes.
- The retriever compares the query embedding against stored note embeddings to find the closest matches.

## Cosine Similarity

- Cosine similarity measures the angle between two vectors.
- It is useful because it focuses on directional similarity rather than raw magnitude.
- In retrieval systems, higher cosine similarity usually means the content is more semantically related to the query.

## Hybrid Retrieval

- Hybrid retrieval combines keyword retrieval and vector retrieval.
- Keyword retrieval is useful for exact phrases, product names, and jargon.
- Vector retrieval is useful for meaning-based matches when the wording differs.
- Combining both tends to improve recall without relying entirely on one retrieval strategy.

## Practical Tradeoff

- If the system misses relevant documents, increase recall by improving chunking, retrieval settings, or both.
- If the system retrieves too much noise, tighten filters or use better metadata.
