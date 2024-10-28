from langchain_openai import ChatOpenAI
def create_openai_language_model() -> ChatOpenAI:
    """Create and return an instance of the language model."""
    return ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=2000)
