from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generator_agent(
    query,
    context,
    memory
):

    if (
        context is None
        or len(context.strip()) < 30
    ):

        return (
            "I could not retrieve sufficiently "
            "relevant information from the document."
        )

    prompt = f"""
    You are a document-grounded AI
    research assistant.

    Your task:
    Answer the user question using
    the retrieved context as the
    PRIMARY source of information.

    IMPORTANT RULES:
    - Prefer retrieved evidence.
    - Avoid unsupported claims.
    - If some details are unclear,
      answer with the BEST available
      evidence instead of refusing.
    - Keep answers factual and concise.
    - Do NOT behave like a generic chatbot.
    - Do NOT mention lack of context
      unless absolutely necessary.

    Conversation History:
    {memory}

    Retrieved Context:
    {context}

    User Question:
    {query}

    Generate a clear grounded answer.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return answer