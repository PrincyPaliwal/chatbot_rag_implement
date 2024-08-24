
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
import pandas as pd

from typing import Optional

app = FastAPI(debug=True)

origins = [
    "https://dev.d3ewkihoy8hhci.amplifyapp.com/",
    "http://localhost:3000/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    question: str
    prompt: Optional[str] = None


class UpsertRequest(BaseModel):
    q_a: str
    collection_name: str

# Initialize databases and model
def db_def(collection_name, ids=None, doc=None, client=None, embedding_function=None):
    if client is None:
        # os.remove("chroma")
        
        client = chromadb.PersistentClient()
        # client.reset()
        
    if embedding_function is None:
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs = {'device': 'cpu'})
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    if ids is not None:
        collection.add(ids=ids, documents=doc)
    langchain_chroma = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    return langchain_chroma

def upsert(q_a, collection_name, client, embedding_function):

    collection = client.get_collection(name=collection_name)
    collection.add(ids=[str(collection.count() + 1)], documents=[q_a])
    langchain_chroma = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    return langchain_chroma

def answer(question,prompt=None, question_db_threshold=1.8, qa_pair_db_threshold=0.7):
    # Step 1: Initialize the language model
    llm = ChatCohere(cohere_api_key="yEknIO6c0in2vwLIkh6JJ9x1WKLke7LPGbfbaiwi")
    # Step 2: Search question_db for similar documents
    question_results = question_db.similarity_search_with_score(question, k=5)
    qa_results = qa_pair_db.similarity_search_with_score(question, k=5)

    

    

    if question_results[0][1]<question_db_threshold:
        if qa_results[0][1]<qa_pair_db_threshold:
            answer = qa_results[0][0].page_content
            print(answer)
            try:
                answer_int = answer.index('\n')
                answer = answer[answer_int+1:]
            except:
                pass
        else:
            upsert(question, 'quest_db',client=client,embedding_function=embedding_function)
            context = ""
            for i in range(5):
                context += qa_results[i][0].page_content + " \n "
            # prompt = f'''
            # Answer the following question based on the context provided or else reply unfortunately i don't know about this and remember you were trained by AI labs at EHR Logic:
            # answer the basic responses like Hey hello or anything accordingly
            # Context: {context.strip()} 

            # Question: {question}
            # '''
            if prompt is None:
                prompt = f'''
                Answer the following question based on the context provided or else reply unfortunately i don't know about this and remember you were trained by AI labs at EHR Logic:
                answer the basic responses like Hey hello or anything accordingly but don't reply except that
                Context: {context.strip()} 

                Question: {question}
                '''
            message = [
                HumanMessage(
                    content = prompt.format(context , question)
                )
            ]
            answer = llm.invoke(message).content
            upsert(f"{question}\n{answer}", 'qa_pair',client=client,embedding_function=embedding_function)
    else:
        answer = "Unfortunately, I don't know about this, I was not trained for this purpose."
    
    return answer
def reset_db():

    df = pd.read_csv("merged_text.csv")
    df1 = pd.read_csv("output.csv")
    if 'text' in df.columns and 'text' in df1.columns:
        concatenated_df = pd.concat([df, df1], ignore_index=True)
    else:
        raise ValueError("Both DataFrames must have a 'text' column")
    
    qa_pair_db = db_def('qa_pair', [str(i) for i in range(1, len(concatenated_df) + 1)], [doc for doc in concatenated_df['text']])
    docs = [concatenated_df['text'][i].strip().split('\n')[0] for i in range(len(concatenated_df))]
    question_db = db_def('quest_db', [str(i) for i in range(1, len(concatenated_df) + 1)], docs)


@app.post("/upsert")
def upsert_endpoint(request: UpsertRequest):
    try:
        upsert(request.q_a, request.collection_name,client,embedding_function)
        return {"message": "Document upserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer")
def answer_endpoint(request: QARequest):
    try:
        response = answer(request.question,request.prompt)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_db")
def reset_db_endpoint():
    try:
        reset_db()
        return {"message": "Database reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    client = chromadb.PersistentClient()
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs = {'device': 'cpu'})
    qa_pair_db = db_def('qa_pair',client=client,embedding_function=embedding_function)
    question_db = db_def('quest_db',client=client,embedding_function=embedding_function)

    uvicorn.run(app, host="0.0.0.0", port=8000)
