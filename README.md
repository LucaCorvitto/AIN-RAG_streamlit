# AIN-RAG: Artificial Intelligence Norms Retrieval-Augmented Generation System
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-0099CC?logo=pinboard&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-FFD700?logo=langchain&logoColor=black)

AIN-RAG is a Retrieval-Augmented Generation (RAG) application designed to help users navigate AI regulatory frameworks. This system integrates Large Language Models (LLMs) with external knowledge sources to provide accurate, context-aware responses enriched with references to the original documents.

---

## Try the demo directly on streamlit cloud!

Start [chatting](https://ain-rag.streamlit.app/) with the model to explore its functionalities!

---

## Why RAG?

Traditional LLMs, though powerful, often face limitations such as hallucination (fabricating information) and lack of updated domain knowledge. Retrieval-Augmented Generation mitigates these challenges by:

1. **Grounding Responses in Data:** RAG systems retrieve relevant chunks of information from an external document database, ensuring that answers are factually accurate and contextually relevant.
2. **Dynamic Knowledge Updates:** Unlike static LLMs, RAG systems can leverage real-time updates to external datasets, making them ideal for domains like AI regulations, which evolve over time.
3. **Improved Explainability:** By providing citations and linking responses to source documents, RAG enhances user trust and transparency.

---

## How It Works

AIN-RAG uses the following workflow, inspired by the [auto-evaluative-rag pipeline](https://instill.tech/george_strong/pipelines/auto-evaluative-rag/preview) by [Instill](https://www.instill.tech/):

1. **User Query:** A user provides a question or prompt.
2. **Query Refinement:** The system revises the query using an LLM to optimize document retrieval.
3. **Document Retrieval:** Relevant chunks of text are retrieved from a (pinecone) vector database using similarity search.
4. **Evaluation:** The system evaluates the relevance, context precision, and faithfulness of the retrieved chunks.
5. **Response Generation:** A response is generated by the LLM, augmented with references to the retrieved chunks.
6. **Metrics Evaluation:** The response is evaluated for quality using predefined metrics.
7. **Interactive Sidebar:** Additional information, including citations, evaluation metrics, and retrieved document chunks, is displayed for transparency and exploration.
 
---

## Features

- **Dynamic Database Selection:** Choose between different regulatory frameworks (e.g., EU or US).
- **Interactive Chat Interface:** Powered by Streamlit, the system provides a seamless chat-based interaction.
- **Citations and Chunk Exploration:** Responses include links to the relevant chunks, ensuring transparency.
- **Metrics Dashboard:** Evaluation metrics like answer relevancy, context precision, and faithfulness are visualized for user insight.

---

## How to Replicate

To replicate the functionality of AIN-RAG for your own domain:

1. **Document Preparation:**
   - Gather documents relevant to your use case.

2. **Database Setup:**
   - Create your own Pinecone index and configure it with appropriate dimensions to match your embeddings.
   - Change the embedding dimension in the `pinecone_embedding.py` and `utils.py` files accordingly.
   - Define the Pinecone index name and set the pinecone API key as an environment variable.

3. **Integrate LLM:**
   - Connect an LLM (e.g., OpenAI's GPT-4o-mini) for query refinement and response generation.
   - You can create an [openai API key](https://platform.openai.com/api-keys) or change the code to allow for free access models (working on it).

4. **Customize Retrieval Logic:**
   - Adapt the retrieval workflow in `retrieve_documents` to suit your dataset and embeddings.

5. **Evaluation Metrics:**
   - Implement metrics to evaluate the generated responses for relevance, precision, and faithfulness or add new ones.

6. **Streamlit Interface:**
   - Use Streamlit components for a user-friendly chat and dashboard interface.

---

## File Structure

- `GUI.py`: Main application script.
- `utils.py`: Mostly langchain functions for query refinement, document retrieval, and response generation.
- `pinecone_embedding.py`: Execute the pinecone indexing of the specified documents.
- `prompts.yaml`: Contains all the prompts used for the different agents.
- `markdown_links.json`: A JSON file mapping filenames to external links.
- `requirements.txt`: Dependency list.
