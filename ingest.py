import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

# create clients
local_client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
pc = Pinecone(api_key = pinecone_api_key)

# creating vector database

index_name = "qwen-rag-chatbot"

if not pc.has_index(index_name):
      pc.create_index(
            name = index_name,
            dimension = 1024,
            metric = "cosine",
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
      )

index = pc.Index(index_name)

#reading and chunking the document

with open("content.txt", "r") as file:
      full_text = file.read()

def chunk_text(text , size = 500):
      chunks = []

      for i in range(0 , len(text), size):
            chunk = text[i:i+size]
            chunks.append(chunk)

      return chunks

chunks = chunk_text(full_text)

# Embedding and Upserting (Ingestion)

for i , text_chunk in enumerate(chunks):
      response = local_client.embeddings.create(
            model= "qwen3-embedding:0.6b",
            input= text_chunk,
      )

      vector = response.data[0].embedding
      index.upsert(vectors = [{
            "id": f"chunk_{i}",
            "values": vector,
            "metadata": {
                  "text": text_chunk
            }
      }])

print("Ingestion Complete! Ready to chat")