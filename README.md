# 🤖 Deep Learning Chatbot (RAG)

This chatbot answers questions about **Deep Learning** using a lightweight **Retrieval-Augmented Generation (RAG)** pipeline. It fetches relevant Wikipedia text and generates answers using a language model.

🔗 **Try it live:** [Deep Learning Chatbot on Hugging Face 🚀](https://huggingface.co/spaces/jagratichauhan15/chatbot)

### 🔍 How It Works

1. **Wikipedia Retrieval**: On startup, the app fetches a summary of the topic “Deep Learning” from Wikipedia.
2. **FAISS Indexing**: The text is chunked and indexed using FAISS for fast retrieval.
3. **Question Answering**: At runtime, user queries are matched to relevant chunks, and passed to a transformer model to generate answers.

### 🧠 Technologies

- `sentence-transformers` for embedding chunks
- `faiss-cpu` for similarity search
- `transformers` + `google/flan-t5-base` for QA generation
- `gradio` for the web interface
- `wikipedia` to fetch initial context

### 🚀 Usage

Click the “Submit” button after typing a question related to deep learning.

Example questions:
- What is deep learning?
- What is a neural network used for?
- What is backpropagation?

### 📦 Requirements
* gradio
* transformers
* torch
* sentence-transformers
* faiss-cpu
* wikipedia

