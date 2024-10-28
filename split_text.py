from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


def recursive_splitter(chunk_size, chunk_overlap):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter

def character_splitter(chunk_size, chunk_overlap):
    text_splitter = CharacterTextSplitter(separator='.', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter

def semantic_splitter(embeddings, breakpoint_threshold_type="percentile"):
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type)
    return text_splitter

def split_documents(splitter_type, documents, **kwargs):
    """
    Splits documents using the specified splitter type.
    
    Parameters:
        splitter_type (str): The type of splitter to use ('recursive', 'character', or 'semantic').
        documents (list): The list of documents to split.
        **kwargs: Additional keyword arguments specific to each splitter function.
    
    Returns:
        list: The split documents.
    """
    splitter_function = SPLITTER_FUNCTIONS.get(splitter_type)
    
    # Validate splitter type
    if splitter_function is None:
        raise ValueError(f"Invalid splitter type '{splitter_type}'. Must be one of {list(SPLITTER_FUNCTIONS.keys())}.")

    # Call the appropriate splitter function with the provided arguments
    text_splitter =  splitter_function(**kwargs)
    texts = text_splitter.split_documents(documents)
    return texts

# Mapping dictionary to associate each splitter name with its respective function.
SPLITTER_FUNCTIONS = {
    "recursive": recursive_splitter,
    "character": character_splitter,
    "semantic": semantic_splitter,
}
