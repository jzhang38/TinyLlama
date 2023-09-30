import json
from tqdm import tqdm
from langchain.retrievers import WikipediaRetriever
retriever = WikipediaRetriever()

import wikipedia


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


if __name__ == '__main__':
	instances = load_data('/data/rag_tinyllama/nq_data/test.jsonl')
	print(f'\nLoad {len(instances)} instances.\n')

	new_instances = []
	for inst in tqdm(instances[:]):
		question, ans_lst = inst['question'], inst['answers']
		retrieved_docs = retrieve_doc(question)
		new_instances.append({
			'question': question,
			'answer': ans_lst,
			'docs': retrieved_docs,
			})

	with open('/data/rag_tinyllama/nq_data/test_doc.jsonl', 'w') as f:
		for entry in new_instances:
			json.dump(entry, f)
			f.write('\n')
