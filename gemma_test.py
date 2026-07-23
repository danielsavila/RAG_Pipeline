from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32
)

prompt = "what is my company's remote work policy"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample = True,
    temperature = .7,
    top_p = .7,
    repetition_penalty = 1.4,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
