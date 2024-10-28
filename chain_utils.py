from langchain_core.pydantic_v1 import BaseModel, Field

class QuestionAnswerFromContext(BaseModel):
    """
    Model to generate an answer to a query based on a given context.
    
    Attributes:
        answer_based_on_content (str): The generated answer based on the context.
    """
    # prompt level
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def build_chain_with_structured_output(question_answer_from_context_prompt, llm):
    # Create a chain by combining the prompt template and the language model
    complex_chain = question_answer_from_context_prompt | llm.with_structured_output(
        QuestionAnswerFromContext)
    return complex_chain

def build_chain(prompt, llm):
    # Create a chain by combining the prompt template and the language model
    simple_chain = prompt | llm
    return simple_chain