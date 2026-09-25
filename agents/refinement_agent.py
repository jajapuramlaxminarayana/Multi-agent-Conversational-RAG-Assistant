from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def refinement_agent(
    query,
    context,
    previous_answer
):

    reflection_prompt = f"""
    You are a hallucination detection
    and correction agent.

    Your task is to critically analyze
    the generated answer.

    Identify:
    1. Unsupported claims
    2. Hallucinated information
    3. Assumptions not present
       in retrieved context
    4. Missing evidence

    User Question:
    {query}

    Retrieved Context:
    {context}

    Previous Generated Answer:
    {previous_answer}

    First:
    - Analyze problems in the answer.
    - Identify unsupported claims.

    Then:
    - Generate a NEW answer STRICTLY
      grounded in retrieved context.

    IMPORTANT RULES:
    - Do NOT infer missing information.
    - Do NOT assume architecture details.
    - Do NOT add external knowledge.
    - Only use retrieved evidence.
    - If information is insufficient,
      explicitly say:
      "The retrieved context does not
      contain enough information."

    Keep the answer:
    - factual
    - concise
    - evidence-grounded

    Output format:

    ANALYSIS:
    <analysis>

    IMPROVED ANSWER:
    <new_answer>
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": reflection_prompt
            }
        ],
        temperature=0.1
    )

    output = (
        response
        .choices[0]
        .message
        .content
    )

    if "IMPROVED ANSWER:" in output:

        improved_answer = output.split(
            "IMPROVED ANSWER:"
        )[-1].strip()

    else:

        improved_answer = output

    return improved_answer