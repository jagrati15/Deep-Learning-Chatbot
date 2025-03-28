# Deep Learning Chatbot (RAG)

This is a Deep Learning-powered chatbot built using Retrieval-Augmented Generation (RAG) architecture. The bot provides answers to questions related to deep learning concepts, such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformers, by retrieving relevant information from Wikipedia articles and generating detailed answers.

## Features

### Retrieval-Augmented Generation (RAG)
- Uses a combination of a retrieval model (FAISS) and a generative model (Hugging Face) to provide contextually relevant answers.

### Deep Learning Concepts
- The bot is trained to respond to queries about deep learning topics like CNNs, RNNs, Transformers, Dropout, and more.

### Wikipedia Integration
- Retrieves information from Wikipedia articles to provide accurate and updated answers.

### Interactive Chat
- The user can interact with the chatbot by asking questions, and the chatbot will respond with answers.

### Streamlit Interface
- A simple web interface to interact with the chatbot.

## How It Works

1. **User Input**: The user submits a question.
2. **Context Retrieval**: The chatbot searches relevant Wikipedia articles and retrieves related content using the FAISS index.
3. **Answer Generation**: A fine-tuned generative model (Phi-1.5 from Hugging Face) generates a detailed and context-aware response.
4. **Chat History**: The chatbot keeps a history of interactions, which can be accessed from the sidebar.

## Requirements

Make sure you have the following dependencies installed:

```bash
streamlit
transformers
torch
sentence-transformers
faiss-cpu
wikipedia-api
