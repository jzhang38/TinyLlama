from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


model_id = "huggyllama/llama-13b"
peft_model_id = "timdettmers/guanaco-13b"
assistant_checkpoint = "PY007/TinyLlama-1.1B-Chat-v0.1"


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)


prompt = "Give me detailed info about Jeo Biden."
formatted_prompt = f"### Human: {prompt}### Assistant:"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)
model.load_adapter(peft_model_id)
print("Large model loaded")
model.config.use_cache = True
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).half().to(device)  
assistant_model.config.use_cache = True
print("Small model loaded")


print("###Native Decoding Starts...\n")
start = time.time()
outputs = model.generate(**inputs, assistant_model=None, max_new_tokens=512)
end = time.time()
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
print("Time: ", end - start)

print("###TinyLlama Assisted Decoding Starts...\n")
start = time.time()
outputs = model.generate(**inputs, assistant_model=assistant_model,max_new_tokens=512)
end = time.time()
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# print time in seconds
print("Time: ", end - start)

