import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 🔐 Load Hugging Face token from Streamlit Secrets or environment variable
huggingface_token = st.secrets.get("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

@st.cache_resource
def load_pipeline():
    model_name = "microsoft/phi-1_5"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)
    
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the model pipeline
hf_pipeline = load_pipeline()

# 🌐 Streamlit App Interface
st.title("🤖 Deep Learning Chatbot")
st.markdown("Ask a question about deep learning. The chatbot will try to answer using the `phi-1_5` model!")

user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Generating answer..."):
        result = hf_pipeline(user_input, max_new_tokens=100, do_sample=True)[0]['generated_text']
        st.markdown(f"**Response:**\n\n{result}")
