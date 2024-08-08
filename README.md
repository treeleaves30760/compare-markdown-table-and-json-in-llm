# Compare markdown table and other table-like data structures with pure text

This project aim to compare the performance of table-like data structure with RAG in llm.

## Usage

You can create a agent from local or api

- local

```python
from interact_local import LLMInference

LLM = LLMInference("meta-llama/Meta-Llama-3-8B-Instruct")
```

- api

```python
from interact_api import LLMInference_api

LLAMA_SERVER = "http://localhost:8081/completion"

LLM = LLMInference_api()
```
