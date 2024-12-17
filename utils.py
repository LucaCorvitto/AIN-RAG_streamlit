import os
import yaml
import streamlit as st
from getpass import getpass
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser

#os.environ['PINECONE_API_KEY'] = "pcsk_288X2e_M4zbjKrEAnunFUJcRYJ5twDDCTfhD1ybW8gsUSkKuZkfpPbGF3dn4WeQgjfAwx8"

# Define the structured metrics model
class Metrics(BaseModel):
    context_precision: float = Field(description="The precision of the context provided.")
    answer_relevancy: float = Field(description="The relevancy of the answer to the context.")
    faithfulness: float = Field(description="How faithful the answer is to the context.")

# Load YAML file
with open("prompts.yaml", "r") as f:
    global PROMPTS
    PROMPTS = yaml.safe_load(f)

def initialization(database='eu'):

    ### Initialize OpenAI Embeddings (for query embedding) ###
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")

    ### Initialize OpenAI Chat Model ###
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    model_name = 'text-embedding-3-small'  # Same model used for document embeddings
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY, dimensions=1536)

    ### Connect to Pinecone ###
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "streamlit-rag" if database=="eu" else "streamlit-rag-us"
    index = pc.Index(index_name)

    ### Initialize the Pinecone Vector Store ###
    vector_store = PineconeVectorStore(index=index, embedding=embed)

    return llm, vector_store

### LANGCHAIN WORKFLOW

def revise_query(model, user_query, system_message=PROMPTS["revise_query_system"]):
    prompt = ChatPromptTemplate([
        ("system", (system_message) ),
    ])
    chain = prompt | model
    revised_query = chain.invoke({"user_query":user_query})

    return revised_query.content

def retrieve_documents(vector_store, revised_query, top_k=5):
    relevant_chunks = vector_store.similarity_search(query=revised_query, k=top_k)
    return relevant_chunks

def generate_response(model, user_query, chat_history, document_chunks, context_precision, system_message=PROMPTS["generate_response_system"], human_message=PROMPTS["generate_response"]):
    prompt = ChatPromptTemplate([
        ("system", (system_message) ),
        ("placeholder", "{messages}"),
        ("human", human_message),
    ])

    chain = prompt | model

    res_chunks = chain.invoke({
        "context":document_chunks,
        "messages":chat_history,
        "context_precision":context_precision,
        "user_query":user_query,
    })

    return res_chunks.content

def evaluate_context_precision(model, user_query, document_chunks, system_message=PROMPTS["evaluate_context_precision_system"]):
    prompt = ChatPromptTemplate([
        ("system", (system_message) ),
    ])

    chain = prompt | model 

    evaluation = chain.invoke({
        "user_query":user_query,
        "document_chunks":document_chunks,
    })

    return evaluation.content

def evaluate_answer_relevancy(model, user_query, response, system_message=PROMPTS["evaluate_answer_relevancy_system"]):
    prompt = ChatPromptTemplate([
        ("system", (system_message) ),
    ])

    chain = prompt | model 

    evaluation = chain.invoke({
        "response":response,
        "user_query":user_query,
    })

    return evaluation.content

def evaluate_faithfulness(model, response, document_chunks, system_message=PROMPTS["evaluate_faithfulness_system"]):
    prompt = ChatPromptTemplate([
        ("system", (system_message) ),
    ])

    chain = prompt | model 

    evaluation = chain.invoke({
        "response":response,
        "document_chunks":document_chunks,
    })

    return evaluation.content

def calculate_metrics(model, context_precision, relevancy_score, faithfulness_score, parser, system_message=PROMPTS["calculate_metrics_system"]):
    prompt = ChatPromptTemplate([
        ("system", (system_message) ), # Inject parser's format instructions
    ]
    )

    chain = prompt | model | parser

    metrics = chain.invoke({
        "context_precision":context_precision,
        "answer_relevancy":relevancy_score,
        "faithfulness":faithfulness_score,
        "format_instructions":parser.get_format_instructions(),
    })

    return metrics

class ListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        """
        Extracts the list from the LLM output.
        Args:
            text (str): The raw response from the LLM.
        Returns:
            list: The extracted list of metrics.
        """
        # Use a regex to find the list of numbers
        import re
        match = re.search(r'\[([0-9.,\s]+)\]', text)
        if match:
            # Convert the string to a Python list
            return [float(x.strip()) for x in match.group(1).split(",")]
        raise ValueError("No list found in the response!")


def response_workflow(user_query, document_chunks, context_precision):
    # Initialization
    model, _ = initialization()
    chat_history = create_chat_history()

    # Generate Response
    res_chunks = generate_response(model, user_query, chat_history, document_chunks, context_precision)

    return res_chunks

def metrics_workflow(user_query, response, document_chunks, context_precision):
    # Initialization
    model, _ = initialization()
    metrics_parser = PydanticOutputParser(pydantic_object=Metrics)

    # Evaluate Answer Relevancy
    relevancy_score = evaluate_answer_relevancy(model, user_query, response)

    # Evaluate Faithfulness
    faithfulness_score = evaluate_faithfulness(model, response, document_chunks)

    # Calculate Metrics
    metrics = calculate_metrics(model, context_precision, relevancy_score, faithfulness_score, metrics_parser)

    return relevancy_score, faithfulness_score, metrics

def create_chat_history():
    message_history = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            message_history.append(HumanMessage(content=dict_message["content"]))
        else:
            message_history.append(AIMessage(content=dict_message["content"]))
        
    return message_history