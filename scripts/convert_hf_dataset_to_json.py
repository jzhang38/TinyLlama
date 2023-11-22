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


if __name__ == '__main__':
    # TODO. Support more datasets(SlimOrca, Tulu v2, UltraChat, etc) and add commond line arguments to specify the dataset.
    new_dataset = process_oasst1_top1_2023()
    if not os.path.isdir('data/raw_json'):
        os.makedirs('data/raw_json')
    with open('data/raw_json/open_assistant.json', 'w') as output_file:
        json.dump(new_dataset, output_file, indent=4)