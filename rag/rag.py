from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.retrievers import WikipediaRetriever
retriever = WikipediaRetriever()
import re
from evaluation import f1, normalize_answer
import json
from tqdm import tqdm


def load_data(fname):
	instance_lst = []
	with open(fname, 'r') as f:
		for line in f:
			instance = json.loads(line)
			instance_lst.append(instance)
	return instance_lst


def retrieve_doc(query):
	return_lst = []
	docs = retriever.get_relevant_documents(query=query)
	if len(docs) > 0:
		for d in docs:
			title = d.metadata['title']
			summary = d.metadata['summary']
			return_lst.append(f'Title: {title}. Summary: {summary}')
	return return_lst


def build_input_str(documents, question):
	templates = [
		'\n'.join(documents),
		'### Human:',
		'Answer this question:',
		question,
		'### Assistant:\n',
	]
	return '\n'.join(templates)


def evaluate_one(predict_seq, gold_ans):
	ans_split = gold_ans.split()
	ans_len = len(ans_split)
	output_split = predict_seq.split()
	output_split_pad = output_split + ['[PAD]'] * ans_len
	f1_score = 0
	for i in range(len(output_split)):
		out_window = output_split_pad[i:i+ans_len]
		f1_score_temp = f1(' '.join(out_window), gold_ans, normalize_answer)
		if f1_score_temp > f1_score:
			f1_score = f1_score_temp
	return f1_score


if __name__ == '__main__':
	# load data
	instances = load_data('/data/rag_tinyllama/nq_data/test_doc.jsonl')
	print(f'\nLoad {len(instances)} instances for evaluation.')

	# exit()

	# load model
	model_name = 'tinyllama-tf'
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

	f1_scores = []
	em = 0
	for inst in tqdm(instances[:100]):
		question, ans_lst = inst['question'], inst['answer']
		
		# Retrieve documents
		# retrieved_docs = retrieve_doc(question)
		retrieved_docs = inst['docs']
		# retrieved_docs = []

		# build inputs
		input_str = build_input_str(retrieved_docs, question)
		input_ids = tokenizer(input_str, return_tensors='pt')['input_ids'].cuda()
		
		output_ids = model.generate(input_ids, max_new_tokens=50)
		output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
		output_str = output_str.split('###')[2]

		# evaluate
		f1_score = 0
		for ans in ans_lst:
			f1_score_t = evaluate_one(output_str, ans)
			if f1_score_t > f1_score:
				f1_score = f1_score_t
		f1_scores.append(f1_score)
		if abs(f1_score_t-1) < 1e-4:
			em += 1

	print('Finish evaluation...')
	print(f'F1 score: {sum(f1_scores) / len(f1_scores):.3f}')
	print(f'Exact match: {em / len(f1_scores)}, {em} / {len(f1_scores)}')








