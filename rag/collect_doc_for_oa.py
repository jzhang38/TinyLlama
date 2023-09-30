import json
from tqdm import tqdm
from langchain.retrievers import WikipediaRetriever

import wikipedia
from datasets import load_dataset
import re
from langdetect import detect

def load_data(fname):
	instance_lst = []
	with open(fname, 'r') as f:
		for line in f:
			instance = json.loads(line)
			instance_lst.append(instance)
	return instance_lst


def retrieve_doc(query, retriever):
	# Now assume only select top documents with first 100 words
	return_lst = []
	docs = retriever.get_relevant_documents(query=query)
	if len(docs) > 0:
		for d in docs:
			title = d.metadata['title']
			summary = d.metadata['summary']
			return_lst.append(f'Title: {title}. Summary: {summary}')
	return return_lst


def make_new_data(data, retrieved_docs):
	text = data['text']
	doc_prefix = '<|im_start|>doc'
	doc_suffix = '<|im_end|>'

	doc_added = []
	for doc in retrieved_docs:
		doc_added.append(doc_prefix)
		doc_added.append(doc + doc_suffix)
		break

	new_data = {
		'text': '\n'.join(doc_added) + '\n' + text
	}
	return new_data


if __name__ == '__main__':
	data_lst = load_dataset('OpenAssistant/oasst_top1_2023-08-25')
	new_data_lst = []
	for d in data_lst['train']:
		text = d['text']
		assert text.startswith('<|im_start|>user\n')
		user_question = text.split('<|im_end|>')[0]
		try:
			user_question = user_question.removeprefix('<|im_start|>user\n')
			lang = detect(user_question)
			if lang.startswith('zh'):
				lang = 'zh'
			retriever = WikipediaRetriever(lang=lang)
			docs = retrieve_doc(user_question, retriever)
			if len(docs) == 0:
				new_data_lst.append(d)
			else:
				# make new inputs
				new_d = make_new_data(d, docs)
				new_data_lst.append(new_d)	
		except:
			new_data_lst.append(d)


	with open('/data/tianduo/rag_oasst/train_doc.jsonl', 'w') as f:
		for entry in new_data_lst:
			json.dump(entry, f)
			f.write('\n')

	# upload to huggingface
	# dataset = load_dataset('/data/tianduo/rag_oasst/')
	# dataset.push_to_hub('Tianduo/rag_oasst')

	# Test
	# dataset = load_dataset('Tianduo/rag_oasst')
	# print(dataset['train'][4]['text'])






