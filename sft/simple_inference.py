

from transformers import AutoTokenizer
import transformers 
import torch
model = "/data/TinyLlama/out/TinyLlama-1.1B-900B-code-preview"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Quick Sort Implementation in golang"
formatted_prompt = (
    f"### Human: {prompt} ### Assistant:"
)

sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p = 0.9,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=1024,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")