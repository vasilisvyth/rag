from typing import List
from langchain import PromptTemplate
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from model_utils import create_openai_language_model
from metrics import correctness_metric, faithfulness_metric
import sys
import os
from retrieval_utils import retrieve_context_per_question, hyde
from chain_utils import build_chain_with_structured_output

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def create_prompt_template(template, input_variables):
    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=template,
        input_variables=input_variables,
    )
    return question_answer_from_context_prompt

def respond_to_question_using_context(question, context, question_answer_from_context_chain):
    """
    Answer a question using the given context by invoking a chain of reasoning.

    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.

    Returns:
        A dictionary containing the answer, context, and question.
    """
    input_data = {
        "question": question,
        "context": context
    }
   
    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}

def build_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retrieved_documents: List[str]
) -> List[LLMTestCase]:

    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer, # prediction
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]

def create_context(questions, chunks_query_retriever, llm, use_hyde):
    info = []
    # Generate answers and retrieve documents for each question
    for question in questions:
        if use_hyde:
            context, hypothetical_doc = hyde(question, llm, chunks_query_retriever)
        else:
            context = retrieve_context_per_question(question, chunks_query_retriever)
        
        info.append((question,context))
    
    return info

def generate_answers(question_answer_from_context_chain, info):
    generated_answers = []
    retrieved_documents = []
    for question, context in info:
        retrieved_documents.append(context)
        context_string = " ".join(context)
        result = respond_to_question_using_context(question, context_string, question_answer_from_context_chain)
        generated_answers.append(result["answer"])

    return generated_answers, retrieved_documents


def evaluate_rag(chunks_query_retriever, questions, ground_truth_answers, prompt_template, use_hyde) -> None:

    llm = create_openai_language_model()
    question_answer_from_context_chain = build_chain_with_structured_output(prompt_template, llm)

    info = create_context(questions, chunks_query_retriever, llm, use_hyde)
    generated_answers, retrieved_documents = generate_answers(question_answer_from_context_chain, info)
    # Create test cases and evaluate
    test_cases = build_eval_test_cases(questions, ground_truth_answers, generated_answers, retrieved_documents)
    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric, faithfulness_metric]
    )
