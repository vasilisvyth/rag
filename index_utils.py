from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# def create_openai_embeds(key, model):
#     emds = OpenAIEmbeddings(api_key=key,model=model)
#     return emds
def create_openai_embeds( model):
    emds = OpenAIEmbeddings(model=model)
    return emds

def create_index(docs, emds):
    vectorstore = FAISS.from_documents(docs, emds)
    return vectorstore

# After creating the vectorstore
def save_index(vectorstore, path):
    vectorstore.save_local(path)

def load_index(path, emds):
    return FAISS.load_local(path, emds,allow_dangerous_deserialization=True)