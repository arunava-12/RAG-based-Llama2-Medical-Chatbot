LLaMA2 Medical Chatbot – CPU Edition
The LLaMA2 Medical Chatbot is a tool designed to provide medical information by answering user queries using state-of-the-art language models and dynamic document retrieval. Users can upload their own medical PDFs, and the bot will instantly integrate them into its knowledge base — no restart required.

This version runs entirely on CPU, so it works on laptops without GPU support.

📑 Table of Contents
Introduction

Features

Architecture

Prerequisites

Installation

Downloading the Model

Getting Started

Usage

Future Plans

Contributing

License

📌 Introduction
The LLaMA2 Medical Chatbot is a retrieval-augmented generation (RAG) system that:

Uses LLaMA2 for language understanding.

Retrieves relevant passages from uploaded PDFs using FAISS vector search.

Runs completely on CPU for compatibility with most laptops.

✨ Features
✅ Dynamic PDF Knowledge Base – Upload new PDFs anytime, and the bot will instantly use them for answering questions.
✅ Fast Semantic Search – Uses embeddings for context-aware document retrieval.
✅ CPU-Only Compatibility – No GPU required; works on most personal machines.
✅ Web-Based Interface – Simple and interactive UI with Chainlit.

🏗 Architecture
mermaid
Copy
Edit
flowchart TD
    U[User] --> UI[Web Interface - Chainlit]
    UI --> UP[Upload PDF]
    UP -->|Extract Text| PYPDF[PyPDF Loader]
    PYPDF --> EMB[SentenceTransformer Embeddings]
    EMB --> VEC[FAISS Vector Store]
    Q[User Question] --> UI
    UI --> LLMQ[Question Processing - LLaMA2]
    LLMQ --> RET[Retrieve Relevant Docs from FAISS]
    RET --> GEN[Generate Answer with Context]
    GEN --> UI
    UI --> U
Explanation:

User uploads PDFs → Text is extracted → Converted to embeddings → Stored in FAISS.

When the user asks a question, LLaMA2 retrieves relevant document chunks.

The model generates an answer using both the document context and its own knowledge.

📦 Prerequisites
Python 3.8+

Required Python packages (installed via requirements.txt):

langchain

chainlit

sentence-transformers

faiss-cpu

pypdf (for PDF loading)

transformers (for LLaMA2 model)

torch (CPU version)

⚙️ Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/llama2-medical-chatbot.git
cd llama2-medical-chatbot
Create and activate a virtual environment (recommended)

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
📥 Downloading the Model
This project uses the LLaMA 2 7B Chat GGML model from TheBloke on Hugging Face.

After cloning this repo, download the file:

python
Copy
Edit
llama-2-7b-chat.ggmlv3.q4_0.bin
and place it inside the models/ folder.

🚀 Getting Started
Make sure your .env file contains any required keys (e.g., HuggingFace access token if using gated models).

Start the Chainlit app:

bash
Copy
Edit
chainlit run model.py -w
Open the app in your browser at http://localhost:8000.

💡 Usage
Upload PDFs – Drag and drop medical PDFs into the interface.

Ask Questions – Type a question in the chat; the bot retrieves relevant info from your uploaded documents.

Get Contextual Answers – The bot responds with an answer and may include references from your documents.

🔮 Future Plans
📷 Medical Image Q&A – Interpret X-rays, MRIs, and other scans.

⚡ GPU Acceleration Option – Faster response times for large models.

📊 Source Highlighting – Show exactly where in the PDF the answer came from.

🤝 Contributing
Contributions are welcome!

Fork this repo.

Create a branch for your feature.

Submit a pull request with a clear explanation of your changes.

📜 License
This project is licensed under the MIT License.

