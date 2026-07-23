from typing import List, Dict
from pandas.core.common import flatten
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("policy_docs")

def augmented_query():
    # get a query from the user
    user_query = input("what is your query for this bot?")

    # creates an embedding the user query, and then compares that to documents in collection
    # result variable returns top 5 most similar sentences relative to the query
    result = collection.query(query_texts = [user_query], 
                            n_results = 3)
    
    #so now we pull the text from the returned most similar entries in the vdb
    # and create a new query that we pass to the llm with that information
    # with the addition that if it does not know, the llm will return "i dont know, reach out to supervisor"

    context = " | ".join(result["documents"][0])

    #using the context, we take the user query + the context and create a combined query
    
    llm_query = f'''Using the context provided, please answer the users question as clearly, accurately, and in no more than 2 sentences.
                    If the information the user is looking for is not in the context provided, say so, and instruct 
                    the user to reach out to their supervisor for clarification.
                    CONTEXT: {context}, 
                    QUESTION: {user_query}
                '''
    return llm_query

augmented_query()