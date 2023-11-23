import json
from datasets import load_dataset
import os
from tqdm import tqdm


def process_oasst1_top1_2023():
    dataset = load_dataset('OpenAssistant/oasst_top1_2023-08-25')
    train_dataset = dataset['train']
    new_dataset = []
    for d in tqdm(train_dataset):
        multi_turn = d['text'].split('<|im_end|>')
        multi_turn = list(map(lambda x: x.strip(), multi_turn))
        multi_turn = list(filter(lambda x: len(x) > 1, multi_turn))
        conversation = []
        c = 0
        ## empty system prompt
        conversation.append({
                    "role": "system",
                    "content": ""
                })
        for t in multi_turn:
            if c%2 == 0:
                assert t.startswith('<|im_start|>user\n')
                t = t[len('<|im_start|>user\n'):]
                conversation.append({
                    "role": "user",
                    "content": t
                })
            else:
                assert t.startswith('<|im_start|>assistant\n')
                t = t[len('<|im_start|>assistant\n'):]
                conversation.append({
                    "role": "assistant",
                    "content": t
                })
            c += 1
        new_dataset.append(conversation)
    return new_dataset


def process_slim_orca():
    dataset = load_dataset('Open-Orca/SlimOrca')
    train_dataset = dataset['train']
    new_dataset = []
    for d in tqdm(train_dataset):
        assert len(list(d.keys())) == 1 and list(d.keys())[0] == 'conversations'
        conversation = d['conversations']
        role_mapping = {
            'system': 'system',
            'human': 'user',
            'gpt': 'assistant',
        }
        new_conversation = list(map(lambda x: 
            {'role': role_mapping[x['from']], 'content': x['value']}, 
            conversation))
        new_dataset.append(new_conversation)
    return new_dataset


def process_tulu_v2():
    dataset = load_dataset('allenai/tulu-v2-sft-mixture')
    train_dataset = dataset['train']
    new_dataset = []
    for d in tqdm(train_dataset):
        new_dataset.append(d['messages'])
    return new_dataset


def setup(dataset: str = 'open_assistant'):

    if dataset == 'open_assistant': data_lst = process_oasst1_top1_2023()
    elif dataset == 'slim_orca': data_lst = process_slim_orca()
    elif dataset == 'tulu-v2': data_lst = process_tulu_v2()
    else: raise NotImplementedError

    if not os.path.isdir('data/raw_json'):
        os.makedirs('data/raw_json')
    with open(f'data/raw_json/{dataset}.json', 'w') as f:
        for d in data_lst:
            json.dump(d, f)
            f.write('\n')



if __name__ == '__main__':

    from jsonargparse import CLI
    CLI(setup)

    