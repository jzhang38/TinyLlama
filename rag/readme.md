## Retrieval-Augmented Generation (RAG)

Currently we use retrieval-augmented generation to enhance TinyLlama's performance on Knowledge-intensive tasks, e.g. NaturalQuestions and MMLU.

Given a query, we retrieve relevant documents from Wikipedia. The following table shows the preliminary results on 100 examples:

| Model       | Add retrieval-augmented module | F1-score  | Exact Match  |
| ----------- | ---------------                | --------- | -----------  |
| TinyLlama   | No                             | 16.9      | 8.0          |
| TinyLlama   | Yes                            | 23.8      | 13.0         |