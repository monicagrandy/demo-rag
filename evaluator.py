import os
from openai import OpenAI


def score_groundedness(question: str, context: str, answer: str, api_key: str) -> dict:
    """
    Groundedness (Faithfulness): Is the answer supported by the retrieved context?
    Uses LLM-as-Judge with temperature=0 for reproducible scores.

    Returns: {"score": float 0.0-1.0, "rationale": str}
    """
    # 1. Build the judge client (uses api_key arg, not env, so app.py controls the key)
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # 2. Build the judge prompt
    prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT. Check for hallucinations or unsupported claims.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Scoring rubric:
- 1.0 = every claim is directly supported by the context
- 0.5 = most claims are supported, minor details are inferred
- 0.0 = one or more claims contradict or are absent from the context

Respond in exactly this format:
Score: <number between 0.0 and 1.0>
Reasoning: <one sentence explaining your score>"""

    # 3. Invoke the judge
    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30,
        )
        output = response.choices[0].message.content

        # 4. Parse score and rationale
        score = 0.5
        rationale = "Could not parse rationale."
        for line in output.strip().splitlines():
            if line.startswith("Score:"):
                try:
                    score = float(line.replace("Score:", "").strip())
                except ValueError:
                    score = 0.5
            elif line.startswith("Reasoning:"):
                rationale = line.replace("Reasoning:", "").strip()

        # 5. Return structured result
        return {"score": score, "rationale": rationale}

    except Exception as e:
        print(f"⚠ Groundedness evaluation error: {str(e)[:80]}")
        return {"score": 0.5, "rationale": f"Evaluation error: {str(e)[:80]}"}
