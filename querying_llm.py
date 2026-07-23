from augmented_query import augmented_query
from transformers import pipeline


llm_query = augmented_query()

pipe = pipeline(
"text-generation",
model="google/gemma-3-1b-it",
dtype="auto",
device_map="auto",
)

def query_llm(llm_query: list):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": llm_query
                }
            ]
        }
    ]

    outputs = pipe(
        messages,
        max_new_tokens = 100
    )
    return outputs

output = query_llm(llm_query)
output[0]["generated_text"][-1]["content"]