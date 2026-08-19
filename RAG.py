import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def rag_pipeline(pdf_path):
    ### loader ###
    print(f"Reading the file {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    ### chuncking ###
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    ### Vectorization ###
    print("Trasforming text to vectors")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    storage = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = storage.as_retriever()

    ### llm model ###
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    ### AI directions ### 
    system_prompt = (
        "You are an AI financial analyst, use only the given sources to respond to the user's questions.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, chain)

    return rag_chain

if __name__ == "__main__":
    

    rag = rag_pipeline("Metcalf.pdf") 
    
    while True:
        erotisi = input("\nMake a question (or 'exit'): ")
        if erotisi.lower() == 'exit':
            break
            
        response = rag.invoke({"input": erotisi})
        print(f"\nRespose AI: {response['answer']}")