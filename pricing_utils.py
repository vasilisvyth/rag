import tiktoken

# Pricing information based on OpenAI's pricing as of October 2024
EMBEDDING_COST_PER_1M_TOKENS = {
"text-embedding-ada-002": 0.1  # for `text-embedding-ada-002`
}
MODEL_QUERY_COST_PER_1M_TOKENS_INPUT = {"gpt-4o-mini":0.0015}
MODEL_QUERY_COST_PER_1K_TOKENS_OUTPUT = {"gpt-4o-mini":0.002}

def calculate_token_count(text, model_name):
    """Estimate token count using OpenAI's `tiktoken` for the specified model."""
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def estimate_embedding_cost(documents, model_name):
    """Calculate total token count and estimated cost for embeddings."""
    total_tokens = sum(calculate_token_count(doc, model_name) for doc in documents)
    cost = (total_tokens / 1_000_000) * EMBEDDING_COST_PER_1M_TOKENS[model_name]
    return total_tokens, cost

def estimate_query_cost(input_tokens, output_tokens, model_name):
    """Estimate cost for querying the OpenAI language model."""
    input_cost = (input_tokens / 1_000_000) * MODEL_QUERY_COST_PER_1M_TOKENS_INPUT[model_name]
    output_cost = (output_tokens / 1_000_000) * MODEL_QUERY_COST_PER_1K_TOKENS_OUTPUT[model_name]
    total_cost = input_cost + output_cost
    return total_cost