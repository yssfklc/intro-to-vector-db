import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.llms.openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone

print(os.getenv("PINECONE_API_KEY"))


pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

if __name__ == '__main__':
    print('Hello Yusuf, Welcome to the new project')
    loader = TextLoader(file_path='./mediumblog1.txt', encoding= None, autodetect_encoding= True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))

    docsearch = Pinecone.from_documents(docs, embeddings, index_name='medium-blogs-embeddings-index')

    query = "What did the president say about Ketanji Brown Jackson"
    docs = docsearch.similarity_search(query)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type='stuff', retriever=docsearch.as_retriever()
    )
    query='What is a vector DB? Give me a 15 word answer for a beginner'
    result=qa({'query':query})
    print(result)