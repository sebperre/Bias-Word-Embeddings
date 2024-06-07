from sentence_transformers import SentenceTransformer

def e5_embed(query: str, model: str):
    if model not in ['large', 'base', 'small', 'large-instruct']:
        raise ValueError(f'Invalid model name {model}')

    embedder = SentenceTransformer(f'intfloat/multilingual-e5-{model}')
    if model == 'large-instruct':
        task = 'Given a short informative text, retrieve relevant topics'
        query = f'Instruct: {task}\nQuery: {query}'

    embeddings = embedder.encode(sentences=[query], convert_to_tensor=False, normalize_embeddings=True)
    return embeddings

# Example usage (768 dimensions)
embeddings = e5_embed('This is a text I want to embed', model='base')
print(embeddings)