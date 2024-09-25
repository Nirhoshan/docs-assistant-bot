import uvicorn
import os
import re
from contextlib import asynccontextmanager
import tiktoken
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from fastapi import FastAPI, HTTPException, Body
from fastapi.logger import logger
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import HumanMessage


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm, encodings, db, reranker, reranked_retriever = None, None, None, None, None
    app.state.system_prompt = """You are a docbot that help users to understand more about Choreo Product usage by answering
    their questions. Information from docs are given to to you help you answer the questions. If the information from docs
    are not relevant to your answer do not use it to construct your answer. If you don't have enough information to answer,
    refuse politely to answer the question. Do not hallucinate!
    ALWAYS construct your answer with generic names for the components or services
    without using specific names like 'Reading List Service'(use the term 'your service' in this case).
    The information given contains markdown images, bullet-points and tables etc. You can make use of them by adding
    them to the response in markdown format. Make sure answers are structured enough to follow through and descriptive.
    In your answer always give the links of the most relevant doc from which you got the answer. You can use the
    doc_link metadata in the docs you are provided. Do not give fake links of your own. Do not always ask the user to
    refer the docs. Give a comprehensive answer to how to do it or what it is before you direct the user to the docs
    with the links. Don't include steps to sign in to choreo console. User is already in Choreo console while asking you this question. """
    app.state.user_prompt_template = """User's Question: %s

    Information from docs:
    """
    try:
        llm = ChatOpenAI(model="gpt-4o")
        logger.info("Creating LLM instance")
        encodings = tiktoken.encoding_for_model("gpt-4o")
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
        reranker = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"),
                                model="rerank-english-v3.0", top_n=5)
        retriever = db.as_retriever(search_kwargs={"k": 20})
        reranked_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
        logger.info("Creating DB instance")
    finally:
        if not llm:
            raise Exception("Failed to create llm instance")
        app.state.llm = llm
        if not encodings:
            raise Exception("Invalid model name")
        app.state.encodings = encodings
        if not db:
            raise Exception("Failed to create db instance")
        app.state.db = db
        if not reranker:
            raise Exception("Failed to create reranker instance")
        if not reranked_retriever:
            raise Exception("Failed to instantiate reranked retriever")
        app.state.reranked_retriever = reranked_retriever
    yield


app = FastAPI(title="Docs Assistant", version="0.1.0", lifespan=lifespan)


async def get_docs(question):
    try:
        reranked_retriever = app.state.reranked_retriever
        results = await reranked_retriever.ainvoke(question)
        return results
    except Exception as e:
        logger.error(f"Error while retrieving the docs: {e}")
        return []


async def get_chat_prompt(question):
    user_prompt_template, encodings = app.state.user_prompt_template, app.state.encodings
    cleaned_question = re.sub(r"in choreo", "", question, flags=re.IGNORECASE)
    results = await get_docs(cleaned_question)
    chat_prompt = user_prompt_template % cleaned_question
    if len(results) == 0:
        chat_prompt += "No docs found for the question"
    chat_prompt_size = len(encodings.encode(chat_prompt))
    all_doc_prompt = []
    for result in results:
        doc_prompt = "Document: {content: %s, metadata:%a}\n" % (result.page_content, result.metadata)
        chat_prompt_size += len(encodings.encode(doc_prompt))
        if chat_prompt_size > 12000:
            break
        all_doc_prompt.append(doc_prompt)
        chat_prompt += doc_prompt
    return chat_prompt.strip()


async def bulk_response(question):
    llm = app.state.llm
    system_prompt = app.state.system_prompt
    message = await get_chat_prompt(question)
    # print(message)
    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ])
        answer = response.content
    except Exception as e:
        logger.error(f"Error while generating llm response: {e}")
        answer = "Error while processing your request. Please try again later"
        raise HTTPException(status_code=500, detail=answer)
    return answer


@app.post('/chat')
async def chat(question: str = Body(..., embed=True)):
    response = await bulk_response(question)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
