'''
1. need to take in a query
2. retrieve the relevant information from database
3. augment query
4. then generate a response
'''
from vector_database import build_vector_database

def augmenting_query():
    vdb, collection = build_vector_database()

    # get a query from the user
    query = input("what is your query for this bot?")

    # creates an embedding for the user query, and then compares that to documents in collection
    # result variable returns top 5 most similar sentences relative to the query
    result = collection.query(query_texts = [query], 
                            n_results = 3)
    
    #so now we pull the text from the returned most similar entries in the vdb
    # and create a new query that we pass to the llm with that information
    # with the addition that if it does not know, the llm will return "i dont know, reach out to supervisor"

    context = [doc for doc in result["documents"]]
    return context

augmenting_query()