from langchain import PromptTemplate
from chain_utils import build_chain

def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.

    Args:
        question: The question for which to retrieve context and URLs.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of unique URLs from the metadata of the relevant documents.
    """

    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.similarity_search(question)

    # Concatenate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]

    return context

def hyde(question, llm, chunk_query_retriever, chunk_size = 1000):
    hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            the document size has be exactly {chunk_size} characters.""",
        )
    hyde_chain = build_chain(hyde_prompt, llm)

    input_variables = {"query": question, "chunk_size": chunk_size}
    hypothetical_doc = hyde_chain.invoke(input_variables).content

    similar_docs = retrieve_context_per_question(hypothetical_doc, chunk_query_retriever)
    return similar_docs, hypothetical_doc
