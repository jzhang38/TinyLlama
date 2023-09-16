

from transformers import AutoTokenizer
import transformers 
import torch
model = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "What are the values in open source projects?"
formatted_prompt = (
    f"### Human: {prompt} ### Assistant:"
)


sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p = 0.7,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")