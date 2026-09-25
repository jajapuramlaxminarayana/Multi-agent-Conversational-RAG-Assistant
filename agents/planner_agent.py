from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def planner_agent(
    query,
    memory
):

    prompt = f"""
    You are a planning agent.

    Use conversation history to understand context.

    Conversation History:
    {memory}

    Rewrite the user's query into a better
    semantic retrieval query.

    Current User Query:
    {query}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    improved_query = (
        response
        .choices[0]
        .message
        .content
    )

    return improved_query