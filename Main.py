"""### Initial Setup"""
import os
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
import time
import re
import json
import fitz
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import ipywidgets as widgets
from IPython.display import display, clear_output
import urllib.request

"""### Data Loading"""

def load_legal_documents():
    """Load all legal documents without any article-specific handling"""
    corpus = []

    # 1. Load PDFs (generic handling)
    def load_pdf(path):
        doc = fitz.open(path)
        return [page.get_text() for page in doc if page.get_text().strip()]

    # 2. Load JSON datasets
    def load_json_dataset(path):
        with open(path) as f:
            data = json.load(f)
        return [f"Q: {item['question']}\nA: {item['answer']}" for item in data]

    # Load all sources
    corpus.extend(load_pdf("Constitution_English.pdf"))
    corpus.extend(load_json_dataset("constitution_qa.json"))
    corpus.extend(load_json_dataset("crpc_qa.json"))
    corpus.extend(load_json_dataset("ipc_qa.json"))

    print(f"📚 Loaded {len(corpus)} legal documents")
    return corpus

corpus = load_legal_documents()

"""### Embedding Model Setup"""

# Initialize embedding model
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Create FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.cpu().numpy())
print(f"✅ FAISS index created with {index.ntotal} vectors")

"""### LLM Model Setup"""

# Model configuration
model_name = "meta-llama/Llama-2-7b-chat-hf"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=True  # Authentication required
    )
except Exception as e:
    print(f"Error loading with device_map: {e}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        token=True
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
print(f"🚀 Model loaded on {next(model.parameters()).device}")

"""### RAG Chat Function"""

def rag_chat(query: str, top_k: int = 3) -> str:
    """
    Generate a legal response using Retrieval-Augmented Generation.

    Args:
        query: The legal question to answer
        top_k: Number of context chunks to retrieve

    Returns:
        Generated legal answer
    """
    # Clean and normalize the query
    query = re.sub(r'\s+', ' ', query).strip()

    # Retrieve relevant context
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    # Build context from top matches
    context = "\n\n".join([corpus[i] for i in indices[0]])

    # System prompt for legal responses
    system_prompt = """You are an expert Indian legal assistant. Provide accurate, concise answers
    to questions about Indian Constitution, laws, and legal procedures.
    - Only respond in English
    - Be precise and cite relevant laws when possible
    - If unsure, say you don't know rather than speculate"""

    # Format the prompt for LLama2
    prompt = f"""<s>[INST]<<<SYS>>>
    {system_prompt}
    <</SYS>>>

    Context:
    {context}

    Question: {query} [/INST]"""

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1
    )

    # Extract and clean the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("[/INST]")[-1].strip()
    return response

"""### User Interface"""

# Widget setup
input_box = widgets.Textarea(
    placeholder="Type your legal question here...",
    layout=widgets.Layout(width="100%", height="100px")
)
ask_btn = widgets.Button(description="Ask", button_style="primary")
output_area = widgets.Output(layout=widgets.Layout(border="1px solid #ccc"))

# Progress indicator
progress = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    description='Processing:',
    bar_style='info',
    style={'bar_color': '#4CAF50'},
    layout=widgets.Layout(width='100%', visibility='hidden')
)

def handle_query(_):
    """Handle user query and display response"""
    with output_area:
        clear_output()
        query = input_box.value.strip()

        if not query:
            print("⚠️ Please enter a legal question")
            return

        try:
            # Show progress
            progress.layout.visibility = 'visible'
            progress.value = 30
            display(progress)

            print("⚖️ Processing your question...")

            # Get and display response
            response = rag_chat(query)
            progress.value = 90

            print("\nAnswer:\n")
            print(response)

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
        finally:
            progress.value = 100
            time.sleep(0.5)
            progress.layout.visibility = 'hidden'

# Connect button and display UI
ask_btn.on_click(handle_query)
display(widgets.VBox([
    widgets.Label("Indian Legal Assistant"),
    input_box,
    ask_btn,
    progress,
    output_area
]))
