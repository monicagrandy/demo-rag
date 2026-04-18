# Agentic RAG And Evaluation

Sun, 12 Apr 26

## Fine-Tuning vs RAG

- Fine-tuning changes model behavior by updating weights with training data.
- RAG keeps the base model fixed and injects external context at runtime.
- Use RAG when freshness and source grounding matter more than permanent behavior changes.

## Evaluation Metrics

- Retrieval evaluation should track precision at K and recall at K.
- Precision at K answers: of the retrieved results, how many were useful?
- Recall at K answers: of the useful results that existed, how many did we actually retrieve?
- Groundedness checks whether the final answer is supported by the retrieved context.

## ReAct Loop

- ReAct stands for a cycle of reason, act, observe, and repeat.
- In an agentic RAG system, the model decides which tool to call, reviews the output, and makes another decision if needed.
- This is useful for questions that require multiple retrieval steps instead of one simple lookup.

## Operational Guidance

- Agentic mode is more flexible than direct RAG, but it is usually slower and harder to debug.
- Use direct RAG first for simple note search and explanation tasks.
