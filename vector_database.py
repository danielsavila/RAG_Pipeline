import chromadb
import os
import spacy
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer

def build_vector_database():
    path = 'C:/Users/daniel.avila/Documents/Github Repos/RCAC/one off projects/RAG_Pipeline/policy repository'
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # first intializde the database
    # that this db is persistent, so files will be saved to disk
    vdb = chromadb.PersistentClient(path="./chroma_db")
    collection = vdb.get_or_create_collection(name = "policy_docs")

    nlp = English()
    nlp.add_pipe('sentencizer')
    for doc in os.listdir(path): 
        doc_name = doc
        # 1. reading text from documents
        doc = nlp(open(os.path.join(path, doc)).read())

        # 2. break apart to sentences and chunk (using sentence chunking from spacy)
        sentences = [sentence.text.strip() for sentence in doc.sents]
        ids = [f'{doc_name}_{i}' for i in range(len(sentences))]

        # 3. create embeddings
        embeddings = model.encode(sentences)

        # 4. store in vector database
        collection.add(embeddings = embeddings, 
                documents = sentences,
                ids = ids,
                metadatas = [{'source': doc_name} for _ in sentences])
                
    return vdb, collection