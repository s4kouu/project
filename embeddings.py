from langchain_huggingface import HuggingFaceEmbeddings # embedding model which converts text into vectors


# custom inherited class from Hugging Face Embeddings for loading model and coding text along with overwitting chosen methods
class E5Embeddings(HuggingFaceEmbeddings):
# method which LangChain invokes whenever parts are being saved to database ( indexing process )
    def embed_documents(self, texts):
        # invokes super().embed_documents from upper class 
        return super().embed_documents([f"passage: {t}" for t in texts]) # list comprehension ( składnia listy ), to each text 't' it adds prefix ( "passage: " )
    # invoked for user's question ( single text )
    def embed_query(self, text):
        return super().embed_query(f"query: {text}") # adds prefix "query: " to the original method
# methods have added prefixes because of e5 which requires them ( this way we modify build-ins only once in code )