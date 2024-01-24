import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms.openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

if __name__ == '__main__':
    print('Hello Yusuf, Welcome to the new project')
    loader = TextLoader("../mediumblogs/ediumblog1.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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