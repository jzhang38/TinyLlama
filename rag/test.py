from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.retrievers import WikipediaRetriever
import re
from evaluation import f1, normalize_answer

# retriever = WikipediaRetriever()
# docs = retriever.get_relevant_documents(query="Who is Barack Obama?")
# print(docs[0].page_content[:300])
# exit()

tokenizer = AutoTokenizer.from_pretrained('tinyllama-tf')
model = AutoModelForCausalLM.from_pretrained('tinyllama-tf').cuda()


def build_input_string(documents, question):
	templates = [
		'\n'.join(documents),
		'### Human:',
		'Answer this question:',
		question,
		'### Assistant:\n',
	]
	return '\n'.join(templates)


def clean_output(output):
	out = output.split('###')

	return out[2]



question = 'what is the smallest prime number that is greater than 30?'
answer = '31'

print(f'The question is:\n{question}\n\n')

# to retrieve docs according to question
documents = [
	# 'Today Tommy wants to go to park. It is because he went to cinema yesterday. Since his father will come tomorrow, he plan to go to supermarket tomorrow.',
]
inputs = build_input_string(documents, question)
inputs = tokenizer(inputs, return_tensors='pt')['input_ids'].cuda()
output_ids = model.generate(inputs, max_new_tokens=50)

outs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print('before retrieval...')
print(clean_output(outs))


if True:
	retriever = WikipediaRetriever()
	docs = retriever.get_relevant_documents(query=question)
	doc_text = docs[0].page_content
	doc_text_split = doc_text.split()
	documents.append(' '.join(doc_text_split[:100]))
	# print(docs[0])
inputs = build_input_string(documents, question)
inputs = tokenizer(inputs, return_tensors='pt')['input_ids'].cuda()
output_ids = model.generate(inputs, max_new_tokens=50)
print('\n\nAfter retrieval...')
outs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
outs = clean_output(outs)
print(outs)

def evaluate_one(prediction_seq, gold_ans):
	ans_split = gold_ans.split()
	ans_len = len(ans_split)
	output_split = prediction_seq.split()
	output_split_pad = output_split + ['[PAD]'] * ans_len
	f1_score = 0
	for i in range(len(output_split)):
		out_window = output_split_pad[i:i+ans_len]
		f1_score_temp = f1(' '.join(out_window), answer, normalize_answer)
		if f1_score_temp > f1_score:
			f1_score = f1_score_temp
	return f1_score

print(evaluate_one(outs, answer))





