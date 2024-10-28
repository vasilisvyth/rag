Execute `run.py` to obtain the RAG results.

### Chunking
We currently support the following chunking methods in `split_text.py`:
- **RecursiveCharacterTextSplitter**
- **CharacterTextSplitter**
- **SemanticChunker**

### Retrieval Stage
In the retrieval stage, we support (see `retrieval_utils.py`):
- Retrieving similar documents to a given query
- Using the Hyde method, which first generates a synthetic document that is then compared with the index

### Evaluation
For evaluation, we also utilize the following objects in `metrics.py`:
- **deepeval** GEval
- **FaithfulnessMetric**
