Data can be found [here](https://drive.google.com/drive/u/0/folders/13n4DS6MMDCl3oM_LnALv-PW-8XTlocmn)

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

### Set Environment Variables:

`pip install python-dotenv`

In the root directory of your project, create a file named .env and add the following variables:

`LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="YOUR_LANGCHAIN_KEY"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="YOUR_PROJECT_NAME"
OPENAI_API_KEY="YOUR_OPENAI_KEY"`

Replace each placeholder (e.g., YOUR_LANGCHAIN_KEY, YOUR_PROJECT_NAME, and YOUR_OPENAI_KEY) with your actual values.

By configuring these environment variables, you can monitor metrics like latency usage, and other performance indicators through the LangSmith dashboard. 