import streamlit as st
import wikipediaapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ First Streamlit command
st.set_page_config(page_title="Deep Learning Chatbot", layout="centered")

# 🔁 Session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# 🔧 Load phi-1_5 securely using Hugging Face token
@st.cache_resource
def load_pipeline():
    huggingface_token = st.secrets["HUGGINGFACE_TOKEN"]
    model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

hf_pipeline = load_pipeline()

# 🌐 Wikipedia setup
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="DeepLearningChatbot/1.0 (https://yourwebsite.com/; your-email@example.com)"
)

# 🧠 Topics
topics = [
    "Convolutional Neural Network",
    "Recurrent Neural Network",
    "Transformer (machine learning)",
    "Dropout (neural networks)",
    "Deep learning"
]

documents = [wiki_wiki.page(topic).text for topic in topics if wiki_wiki.page(topic).exists()]

def split_into_chunks(text, max_words=200):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

chunked_documents = []
for doc in documents:
    chunked_documents.extend(split_into_chunks(doc))

# 🔍 Embeddings + FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(chunked_documents)

dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(document_embeddings))
doc_mapping = {i: chunk for i, chunk in enumerate(chunked_documents)}

# 🔎 Retrieve context
def retrieve_context(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [doc_mapping[idx] for idx in indices[0]]

# 🧠 Generate answer with context
def generate_answer(query, context):
    if not context:
        return "Sorry, I couldn't find enough information to answer your question."

    prompt = (
        f"You are a helpful deep learning assistant. "
        f"Based on the context below, provide a detailed and clear explanation.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )
    result = hf_pipeline(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()

# 💬 Chatbot logic
def chatbot(query):
    q = query.lower().strip()
    if q in ["hello", "hi"]:
        return "Hello! 👋 I'm your deep learning assistant. Ask me anything about CNNs, RNNs, or Transformers!"
    elif q in ["bye", "goodbye"]:
        return "Goodbye! 👋 Come back anytime!"
    elif q in ["thanks", "thank you"]:
        return "You're welcome! 😊"
    else:
        context = " ".join(retrieve_context(query))

        if not context:
            return "Sorry, I couldn't find enough information to generate a meaningful answer."
        return generate_answer(query, context)

# 🎨 UI
st.title("🧠 Deep Learning Chatbot (RAG)")
st.markdown("💬 Ask me anything about deep learning.")

# 🧾 Input using st.text_input
query = st.text_input("🧠 Ask your question:")

if query:
    with st.spinner("🤖 Thinking..."):
        answer = chatbot(query)
    st.success("✅ Answer:")
    st.write(answer)

    # Save to history
    st.session_state.chat.append((query, answer))

# 📜 Sidebar chat history
with st.sidebar:
    st.header("📚 Chat History")
    if st.session_state.chat:
        for i, (q, a) in enumerate(st.session_state.chat):
            st.markdown(f"**{i+1}. Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
    else:
        st.markdown("Ask your first question to get started!")
