from FlagEmbedding import BGEM3FlagModel

def bge_m3_embed(query: str):
    # Can add "use_fp16=True" to speed up predictions
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    embeddings = model.encode([query])['dense_vecs'][0]
    return embeddings

# Example usage (1024 dimensions)
embeddings = bge_m3_embed("This is a text I want to embed")
print(embeddings)