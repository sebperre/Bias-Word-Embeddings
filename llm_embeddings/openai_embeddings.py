from openai import OpenAI
import os

#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
#openai_client = OpenAI(api_key=OPENAI_API_KEY)
openai_client = OpenAI(api_key="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")

def openai_embed(query: str, model="text-embedding-3-large"):
   query = query.replace("\n", " ")
   response = openai_client.embeddings.create(input=[query], model=model)
   embedding = response.data[0].embedding
   return embedding

# Example usage (3072 dimensions)
embeddings = openai_embed("This")
print(embeddings)