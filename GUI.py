import streamlit as st
import os
import time
from collections import defaultdict
import re
import streamlit.components.v1 as components
import pandas as pd
import json
from utils import *

############################################################################################
################################ FUNCTIONS DEFINITION ######################################
############################################################################################

# Load the JSON data (adjust the path as necessary)
with open('markdown_links.json', 'r') as f:
    markdown_links = json.load(f)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    enable_selectbox()

def disable_selectbox():
    st.session_state.disabled = True

def enable_selectbox():
    st.session_state.disabled = False

# Function to replace chunk UIDs with linkable text
def replace_chunk_uid_with_ref(match):
    chunk_uid = match.group(3)  # Capture the core Chunk-Uid
    
    # If it's the first occurrence of this chunk UID, record it
    if chunk_uid not in first_occurrence:
        first_occurrence[chunk_uid] = len(first_occurrence) + 1  # Assign the next available order

    occurrence_order = first_occurrence[chunk_uid]  # Get the order for the current chunk UID
    
    # Return linkable text
    return f'<a href="#chunk-uid:{chunk_uid}">[{occurrence_order}]</a>'

def replace_chunk_uid_with_hyperlink(match):
    chunk_uid = match.group(3)
    # Return linkable text
    return f'<a href="#chunk-uid:{chunk_uid}">{match.group(0)}</a>'

def generate_clickable_link(filename):
    link = markdown_links.get(filename)
    if link:
        return f'<a href="{link}" target="_blank">{filename}</a>'
    return filename  # Return the filename if no link is found

# Add JavaScript to make it scroll to the element
scroll_script = """
<script>
    // Scroll to the element with the ID when the link is clicked
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
</script>
"""

############################################################################################

def write_info_in_sidebar(message):

    # define variables
    chunks = message["info"]["chunks"]
    answer_relevancy = message["info"]["answer_relevancy"]
    context_precision = message["info"]["context_precision"]
    faithfulness = message["info"]["faithfulness"]
    revised_query = message["info"]["revised_query"]
    metrics = message["info"]["metrics"]

    citations = [chunk.metadata['source'] for chunk in chunks]

    # Data for the bar chart
    precision_score = metrics.context_precision*100
    relevancy_score = metrics.answer_relevancy*100
    faithfulness_score = metrics.faithfulness*100

    data = {
        'Metrics': ['Answer Relevancy', 'Context Precision', 'Faithfulness'],
        'Values': [relevancy_score, precision_score, faithfulness_score]  # You can set your own percentages here
    }

    # Creating a DataFrame
    df = pd.DataFrame(data)

    with st.sidebar:

            #st.subheader("Revised Query", divider=True)
            #st.write("This is the query revised processed by the model to retrieve more accurate information from the database:")
            #st.markdown(f"***{revised_query}***")
            #st.text(revised_query)

            # st.divider()

            st.subheader("Citations", divider=True)
            st.write("These are the documents referenced by the model to generate the response:")
            cite_set = set(citations)
            #cite_list = []
            for citation in cite_set:
                citation_link = generate_clickable_link(citation)
                st.markdown(f"- {citation_link}", unsafe_allow_html=True)
            st.markdown(f'<br>', unsafe_allow_html=True)
            st.markdown(f'<a href="#retrieved_chunks">[See more]</a>', unsafe_allow_html=True)

            # st.divider()

            st.subheader("Evaluation Metrics", divider=True)
            st.write("Here is a bar graph showing the evaluation of the response made by other models on 3 different metrics.")
            st.markdown("<br><br>", unsafe_allow_html=True)  # Adds an empty line

            st.bar_chart(df.set_index('Metrics'), color="#2345ad") # #3dabff

            st.markdown(f'<a href="#answer_relevancy">[See more]</a>', unsafe_allow_html=True)
            
            # st.divider()

            st.subheader("Retrieved Chunks", divider=True)
            st.markdown(f'<div id="retrieved_chunks"></div>', unsafe_allow_html=True)
            # Group chunks by source-file-name
            grouped_chunks = defaultdict(list)

            for chunk in chunks:
                grouped_chunks[chunk.metadata["source"]].append((chunk.page_content.strip(), chunk.id))

            # Output grouped chunks avoiding citation repetitions
            for doc, doc_chunks in grouped_chunks.items():
                # Generate a clickable link for the document name
                doc_link = generate_clickable_link(doc)
                st.markdown(f"From {doc_link}:", unsafe_allow_html=True)
                
                # Write each chunk associated with the document
                for chunk_text,chunk_uid in doc_chunks:
                    with st.expander(f"Chunk UID: {chunk_uid}"):
                        st.markdown(f'<div id="chunk-uid:{chunk_uid}"></div>', unsafe_allow_html=True)
                        st.write(chunk_text)
            
            # st.divider()

            st.subheader("Answer Relevancy", divider=True)
            st.markdown(f'<div id="answer_relevancy"></div>', unsafe_allow_html=True)
            st.write(answer_relevancy)


            # st.divider()

            st.subheader("Context Precision", divider=True)
            st.markdown(f'<div id="context_precision"></div>', unsafe_allow_html=True)
            st.markdown(context_precision, unsafe_allow_html=True)

            # st.divider()

            st.subheader("Faithfulness", divider=True)
            st.markdown(f'<div id="faithfulness"></div>', unsafe_allow_html=True)
            st.write(faithfulness)

############################################################################################
####################################### MAIN BODY ##########################################
############################################################################################

# App title
st.set_page_config(page_title="AIN-RAG💬 Artificial Intelligence Norms RAG system")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Initialize Selectbox state
if 'disabled' not in st.session_state:
    st.session_state.disabled = False

# Inject the JavaScript into the page
components.html(scroll_script, height=0)

# Dictionary to store the first occurrence order of unique chunk UIDs
first_occurrence = {}

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

with st.sidebar: # elements in the sidebar
    st.title('AIN-RAG💬 Artificial Intelligence Norms RAG system')

    #st.write('Expand your knowledge about AI regulations with AIN-RAG!')
    
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai_api = st.secrets['OPENAI_API_KEY']
    else:
        openai_api = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai_api.startswith('sk-') and len(openai_api)==164):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['OPENAI_API_KEY'] = openai_api

    # Adjustment of model parameters - choose the database
    database = st.sidebar.selectbox(
        'Select a country', ['EU', 'US'],
         key='database',
         disabled=st.session_state.disabled,
        ).lower()
    
    st.html("<br>")
    
    activate_chat = st.button(
        "Click here to start chatting about the selected regulatory framework!", 
        on_click=disable_selectbox,
        disabled=st.session_state.disabled,
        type="primary")
    
    st.divider()

    st.button('Clear Chat History', 
              on_click=clear_chat_history)

    st.header("Additional Information", divider="rainbow")
    st.markdown("Here you will find additional info to the generated answer, so keep your sidebar **always open**.")


# User-provided prompt
if not st.session_state.disabled:
    prompt = st.chat_input("Select a country and confirm clicking the button below to start chatting.", disabled = True)
else:
    prompt = st.chat_input("Write your question about the AI norms here")
if prompt: #prompt is the user chat input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt) # visualize the user input in the chat

############################################################################################
######################################## QUERY PROCESSING ##################################
############################################################################################

# Processing user query
if st.session_state.messages[-1]["role"] != "assistant":
    if prompt:
        with st.chat_message("assistant"):
            # initialize model and vector store
            model, vector_store = initialization(database)
            # divide the first steps in order to let the user enjoy the processing
            with st.spinner("Retrieving information from documents..."):
                revised_query = revise_query(model, prompt)
                chunks = retrieve_documents(vector_store, revised_query)

            with st.spinner("Checking if your question is in context..."):
                context_precision = evaluate_context_precision(model, prompt, chunks)

            with st.spinner("Answering..."):
                # return Langchain workflow output
                response = response_workflow(prompt, chunks, context_precision)

                # Dictionary to keep track of chunk-uid occurrences
                chunk_uid_occurrences = defaultdict(int)

                # pattern to be replaced
                pattern = r"(\()?(Chunk-Uid[0-9]*)\s*:\s*(<?[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}>?)(\))?"
                pattern_context = r"(\()?(Chunk UID[0-9]*)\s*:\s*(<?[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}>?)(\))?"

                # Substitute chunk-uid to make them clickable
                processed_response_text = re.sub(pattern, replace_chunk_uid_with_ref, response)
                processed_context_precision = re.sub(pattern_context, replace_chunk_uid_with_hyperlink, context_precision)

                # Display the processed response text
                st.write(processed_response_text, unsafe_allow_html=True)

        with st.sidebar:
            with st.spinner("Processing Output Analysis..."):
                answer_relevancy, faithfulness, metrics = metrics_workflow(prompt, response, chunks, context_precision)      
        
        # store and save the outputs in the session
        message = {"role":  "assistant",
                            "content": processed_response_text, 
                    "info": {"chunks":chunks,
                            "revised_query":revised_query,
                            "answer_relevancy":answer_relevancy,
                            "context_precision":context_precision,
                            "faithfulness":faithfulness,
                            "metrics": metrics}
                    } 
        st.session_state.messages.append(message)
        write_info_in_sidebar(message)
