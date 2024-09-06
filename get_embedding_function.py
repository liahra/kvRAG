from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings



# TESTER 
# embedding_function = get_embedding_function()
# # Check if the returned object is an instance of OllamaEmbeddings
# print(f"Is the embedding function an instance of OllamaEmbeddings? {isinstance(embedding_function, OllamaEmbeddings)}")

# # Test the embedding function with some sample text
# sample_text = "Dette er en test!"
# try:
#     embeddings = embedding_function.embed_query(sample_text)
#     print("Embedding generated successfully!")
#     print(embeddings)  # This should print the generated embeddings
# except Exception as e:
#     print(f"Error during embedding generation: {e}")