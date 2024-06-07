import cohere  
co = cohere.Client(api_key="InHwytsfm89CUjCQHybUCn3uIvEwAgvpdMqFkIOI")  
texts = [  
   'Hello from Cohere!', 'مرحبًا من كوهير!', 'Hallo von Cohere!',  
   'Bonjour de Cohere!', '¡Hola desde Cohere!', 'Olá do Cohere!',  
   'Ciao da Cohere!', '您好，来自 Cohere！', 'कोहेरे से नमस्ते!'  
]  
response = co.embed(texts=texts,input_type='classification', embedding_types=['float'], model='embed-multilingual-v3.0')  
embeddings = response.embeddings.float # All text embeddings 
print(embeddings[0]) # Print embeddings for the first text
