import re
import csv
from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset('OpenAssistant/oasst_top1_2023-08-25')
    train_dataset = dataset['train']
    new_dataset = []
    for d in train_dataset:
        multi_turn = d['text'].split('<|im_end|>')
        multi_turn = list(map(lambda x: x.strip(), multi_turn))
        multi_turn = list(filter(lambda x: len(x) > 1, multi_turn))

        temp_lst = []
        c = 0
        for t in multi_turn:
            if c%2 == 0:
                assert t.startswith('<|im_start|>user\n')
                t = t[len('<|im_start|>user'):]
                if c > 1:
                    t = '### Instruction:\n' + t
                temp_lst.append(t)
            else:
                assert t.startswith('<|im_start|>assistant\n')
                t = t[len('<|im_start|>assistant'):]
                if c > 1:
                    t = '\n### Response:' + t
                temp_lst.append(t)
            c += 1
        new_dataset.append({
            'instruction': temp_lst[0],
            'input': '',
            'output': '\n'.join(temp_lst[1:]),
            })

    keys = new_dataset[0].keys()
    with open('data/raw_csv/open_assistant.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(new_dataset)
    
