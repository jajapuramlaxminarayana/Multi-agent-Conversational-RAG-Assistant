from sentence_transformers import (
    SentenceTransformer
)
from sklearn.metrics.pairwise import (
    cosine_similarity
)
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import os
import re

load_dotenv()


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def critic_agent(
    answer,
    context
):

    answer_embedding = embedding_model.encode(
        [answer]
    )

    context_embedding = embedding_model.encode(
        [context]
    )

    similarity = cosine_similarity(
        answer_embedding,
        context_embedding
    )[0][0]

    embedding_score = similarity * 100

    evaluation_prompt = f"""
    You are a hallucination detection agent.

    Evaluate whether the generated answer
    is fully supported by the retrieved context.

    Retrieved Context:
    {context}

    Generated Answer:
    {answer}

    Evaluate:
    1. factual consistency
    2. unsupported claims
    3. hallucinations
    4. missing evidence

    Give:
    - a grounding score from 0 to 100
    - short explanation

    Output STRICTLY in this format:

    SCORE: <number>

    REASON:
    <short explanation>
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": evaluation_prompt
            }
        ],
        temperature=0.1
    )

    evaluation = (
        response
        .choices[0]
        .message
        .content
    )

    match = re.search(
        r"SCORE:\s*(\d+)",
        evaluation
    )

    if match:

        llm_score = float(
            match.group(1)
        )

    else:

        llm_score = 50

    final_score = (
        0.4 * embedding_score
        +
        0.6 * llm_score
    )

    return {
        "score": round(final_score, 2),
        "reason": evaluation
    }