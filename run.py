from langchain.document_loaders import PyPDFLoader
import os
from split_text import recursive_splitter, character_splitter, semantic_splitter, split_documents
from index_utils import create_index, create_openai_embeds, load_index, save_index
import argparse
import yaml
from pathlib import Path
import logging
import json
from pricing_utils import estimate_embedding_cost
import os
import getpass


FAISS_INDEX_NAME = 'faiss_index'
QA_FILENAME = "climate_change_q_a.json"
CONFIGS_FOLDER = 'configs'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

from eval_utils import evaluate_rag, create_prompt_template

def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def read_json(q_a_file_name):
    # Load questions and answers from JSON file
    with open(q_a_file_name, "r", encoding="utf-8") as json_file:
        q_a_json = json.load(json_file)
    return q_a_json

def make_dir_if_not_exists(args):
    if not os.path.exists(args):
        os.makedirs(args)
        logger.info(f"Created directory: {args}")
    else:
        logger.info(f"Directory already exists: {args}")

def load_pdf_documents(pdf_path) -> list:
    """Load documents from the specified PDF file."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def main(config):
    data_path = Path(config['data_path'])
    save_dir = Path(config['save_dir'])

    make_dir_if_not_exists(data_path)
    
    path = data_path / config['pdf_name']
    
    emds = create_openai_embeds(config['embed_model_name'])

    documents = load_pdf_documents(path)

    splitter_type = config['splitter_type']
    split_args = config['splitters'][splitter_type]
    if splitter_type == "semantic":
        split_args['embeddings'] = emds


    docs = split_documents(splitter_type, documents=documents, **split_args)

    docs_content = [doc.page_content for doc in docs]
    tokens, cost = estimate_embedding_cost(docs_content, config['embed_model_name'])
    logger.info(f'tokens {tokens} cost {cost}')


    make_dir_if_not_exists(config['save_dir'])

    index_path = save_dir / f'{FAISS_INDEX_NAME}_{splitter_type}'
    if os.path.exists(index_path):
        index = load_index(index_path, emds)
    else:
        index = create_index(docs, emds)

    save_index(index, index_path)

    index.as_retriever(search_kwargs={"k": config['top_k']})
    q_a_file_name = data_path / QA_FILENAME
    q_a_json = read_json(q_a_file_name)

    questions = [qa["question"] for qa in q_a_json]
    ground_truth_answers = [qa["answer"] for qa in q_a_json]

    if config['num_questions']:
        questions = questions[:config['num_questions']]
        ground_truth_answers = ground_truth_answers[:config['num_questions']]
    
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """
    input_variables=["context", "question"]

    question_answer_from_context_prompt = create_prompt_template(question_answer_prompt_template, input_variables)
    evaluate_rag(index, questions, ground_truth_answers, question_answer_from_context_prompt, config['use_hyde'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a PDF document and create a vector store')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    config_path = os.path.join(CONFIGS_FOLDER,args.config)
    config = load_config(config_path)
    main(config)