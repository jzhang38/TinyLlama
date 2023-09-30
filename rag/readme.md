## Retrieval-Augmented Generation (RAG)

### Prepare evaluation data

```bash
python prepare_data/prepare_qa.py --output_directory /data/rag_tinyllama
```
This will download train, dev, test splits of Natural Questions dataset under /data/rag_tinyllama/nq_data


### Do evaluation

Run the following commands.
```bash
python rag.py
```
Remember to change the model name before evaluation.

It has been shown that the prompt template is very important to the final performance. Please check if you are using a good prompt before perform the evaluation in the rag.py --> build_input_str().


### Preliminary results

Currently we use retrieval-augmented generation to enhance TinyLlama's performance on Knowledge-intensive tasks, e.g., NaturalQuestions and MMLU.

Given a query, we retrieve relevant documents from Wikipedia. The following table shows the preliminary results on 100 examples:

| Model       | Add retrieval-augmented module | F1-score  | Exact Match  |
| ----------- | ---------------                | --------- | -----------  |
| TinyLlama   | No                             | 16.9      | 8.0          |
| TinyLlama   | Yes                            | 23.8      | 13.0         |