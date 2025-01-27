import uvicorn
import os
import re
import logging
from fastapi import FastAPI, Body
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

app = FastAPI(title="Docs Assistant", version="0.1.0")

# Initialize components
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Milvus(
    embedding_function=embeddings,
    collection_name=os.getenv("COLLECTION_NAME"),
    connection_args={
        "uri": os.getenv("ZILLIZ_CLOUD_URI"),
        "token": os.getenv("ZILLIZ_CLOUD_API_KEY"),
        "secure": True,
    },
)
reranker = CohereRerank(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="rerank-english-v3.0",
    top_n=5
)
retriever = db.as_retriever(search_kwargs={"k": 20})
reranked_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)

SYSTEM_PROMPT = """You are a docbot that help users to understand more about Choreo Product usage by answering
their questions. Information from docs are given to to you help you answer the questions. If the information from docs
are not relevant to your answer do not use it to construct your answer. If you don't have enough information to answer,
refuse politely to answer the question. Do not hallucinate!
The information given contains markdown images, bullet-points and tables etc. You can make use of them by adding
them to the response in markdown format. Make sure answers are structured enough to follow through and descriptive.
In your answer always give the links of the most relevant doc from which you got the answer. You can use the
doc_link metadata in the docs you are provided. Do not give fake links of your own. Do not always ask the user to
refer the docs. Give a comprehensive answer to how to do it or what it is before you direct the user to the docs
with the links. Don't include steps to sign in to choreo console. User is already in Choreo console while asking you this question."""

def get_docs(question):
    return reranked_retriever.invoke(question)

def get_chat_prompt(question):
    cleaned_question = re.sub(r"in choreo", "", question, flags=re.IGNORECASE)
    results = get_docs(cleaned_question)
    logging.info(results)
    chat_prompt = f"User's Question: {cleaned_question}\n\nInformation from docs:\n"
    for result in results:
        chat_prompt += f"Document: {{content: {result.page_content}, metadata:{result.metadata}}}\n"
    return chat_prompt.strip()


def bulk_response(question):
    message = get_chat_prompt(question)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ])
    logging.info(response)
    return response.content


@app.post('/chat')
def chat(question: str = Body(..., embed=True)):
    return bulk_response(question)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)