import os
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

local_client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "qwen-rag-chatbot"
index = pc.Index(index_name)

history = [
    {
        "role": "system",
        "content": "you are a helpful assistant. Use the provided context to answer the questions.",
    }
]

while True:
    user_query = input("\n You: ")
    if user_query.lower() in ["quit", "exit"]:
        break

    query_vector = (
        local_client.embeddings.create(model="qwen3-embedding:0.6b", input=user_query)
        .data[0]
        .embedding
    )

    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    context = "\n".join([match.metadata["text"] for match in results.matches])

    augmented_prompt = f"Context:\n {context}\n\n User Question: {user_query}"
    history.append({"role": "user", "content": augmented_prompt})

    # Rolling window to prevent context bloat
    if len(history) > 5:
        history = [history[0]] + history[-4:]

    # Switched to the standard, non-reasoning model
    response = local_client.chat.completions.create(
        model="qwen2.5:3b", messages=history, stream=True
    )

    print("Bot: ", end="", flush=True)
    full_reply = ""

    # Simplified standard streaming loop
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_reply += content

    print("\n")

    history[-1] = {"role": "user", "content": user_query}
    history.append({"role": "assistant", "content": full_reply})