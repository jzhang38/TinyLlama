import json

instance_lst = []
with open('/data/tianduo/atlas_data/data/nq_data/dev.jsonl', 'r') as f:
	for line in f:
		instance = json.loads(line)
		instance_lst.append(instance)


print(instance_lst[4])