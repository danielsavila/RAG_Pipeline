import chromadb
import os
import spacy
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer


path = 'C:/Users/daniel.avila/Documents/Github Repos/RCAC/one off projects/RAG_Pipeline/policy repository'

# 1. load documents
def build_vector_database(path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # first intializde the database
    # note that this db is not persistent, so will be loaded in memory and then lost when program ends
    vdb = chromadb.Client()
    collection = vdb.create_collection(name = "policy_docs")

    # then we...
    # 3. create embeddings
    # 4. store in vector database
    nlp = English()
    nlp.add_pipe('sentencizer')
    for doc in os.listdir(path): 
        doc_name = doc
        doc = nlp(open(os.path.join(path, doc)).read())
        # 2. chunk (using sentence chunking)
        sentences = [sentence.text.strip() for sentence in doc.sents]
        ids = [f'{doc_name}_{i}' for i in range(len(sentences))]
        # 3. create embeddings
        embeddings = model.encode(sentences)
        collection.add(embeddings = embeddings, 
                ids = ids,
                metadatas = [{'source': doc_name} for _ in sentences])
    return vdb, collection