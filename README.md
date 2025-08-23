# 🏥 RAG based LLaMA2 Medical Chatbot – CPU Edition (with PDF + Image Support)

The **LLaMA2 Medical Chatbot** is an intelligent assistant that provides medical information by answering user queries using **state-of-the-art language models** and **dynamic document/image retrieval**.

Users can upload **PDFs or medical images (JPG/PNG/TIFF, etc.)**, and the bot will instantly integrate them into its knowledge base — no restart required.

This version runs entirely on **CPU**, so it works on laptops without GPU support.

---

## 📑 Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Architecture](#architecture)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Downloading the Model](#downloading-the-model)
* [Ingesting Data](#ingesting-data)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Future Plans](#future-plans)
* [Contributing](#contributing)
* [License](#license)

---

## 📌 Introduction

The LLaMA2 Medical Chatbot is a **retrieval-augmented generation (RAG)** system that:

* Uses **LLaMA2 (local CPU)** or **Groq-hosted models** for question answering.
* Retrieves relevant passages from uploaded PDFs/images using **FAISS vector search**.
* Extracts text from images via **OCR (OpenCV + Tesseract)**.
* Runs **completely on CPU** for compatibility with most laptops.

---

## ✨ Features

✅ **Dynamic PDF & Image Knowledge Base** – Upload new PDFs or images anytime, and the bot will instantly use them.  
✅ **Fast Semantic Search** – Uses embeddings for context-aware document retrieval.  
✅ **OCR Image Support** – Extracts medical text from images (e.g., prescriptions, scanned notes).  
✅ **Dual LLM Support** – Choose between **local LLaMA2** or **Groq API** (`groq: your question`).  
✅ **CPU-Only Compatibility** – No GPU required.  
✅ **Web-Based Interface** – Simple, interactive UI with **Chainlit**.

---

## 🏗 Architecture

```mermaid
flowchart TD
    U[User] --> UI[Web Interface - Chainlit]
    UI --> UP[Upload PDF/Image]
    UP -->|PDF| PYPDF[PyPDF Loader]
    UP -->|Image| OCR[OpenCV + Tesseract OCR]
    PYPDF --> EMB[SentenceTransformer Embeddings]
    OCR --> EMB
    EMB --> VEC[FAISS Vector Store]

    Q[User Question] --> UI
    UI --> LLMQ[Question Processing - LLaMA2 or Groq]
    LLMQ --> RET[Retrieve Relevant Docs from FAISS]
    RET --> GEN[Generate Answer with Context]
    GEN --> UI
    UI --> U
````

---

## 📦 Prerequisites

* **Python 3.9+**
* Required Python packages (installed via `requirements.txt`):

  * `langchain`
  * `chainlit`
  * `sentence-transformers`
  * `faiss-cpu`
  * `pypdf` (for PDF loading)
  * `opencv-python` + `pytesseract` (for OCR on images)
  * `transformers`
  * `torch` (CPU version)

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/llama2-medical-chatbot.git
cd llama2-medical-chatbot
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📥 Downloading the Model

This project uses the **LLaMA 2 7B Chat GGML** model from [TheBloke on Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML).

After cloning this repo, download the file:

```
llama-2-7b-chat.ggmlv3.q4_0.bin
```

and place it inside the `models/` folder.

---

## 📂 Ingesting Data

Before starting the chatbot, you can **preload PDFs** into the FAISS vector database:

```bash
py ingest.py
```

You’ll see progress messages like:

```
[INFO] Loaded 3 documents.
[INFO] Split into 120 chunks.
[INFO] Embeddings model loaded.
[SUCCESS] Vector database saved to: vectorstore/db_faiss
```

This ensures your PDFs are ready for semantic search.

---

## 🚀 Getting Started

1. Make sure your `.env` file contains any required keys (e.g., `GROQ_API_KEY` for Groq).
2. Start the Chainlit app:

```bash
chainlit run model.py -w
```

3. Open the app in your browser at **[http://localhost:8000](http://localhost:8000)**.

---

## 💡 Usage

1. **Ask Questions**

   * `local: What is diabetes?` → Uses local LLaMA2 model
   * `groq: What is diabetes?` → Uses Groq API model

2. **Upload PDFs/Images** – Drag and drop medical PDFs or images.

3. **Get Contextual Answers** – The bot responds with an answer, optionally including sources.

---

## 🔮 Future Plans

* 📷 **Advanced Medical Image Q\&A** – Interpret X-rays, MRIs, etc.
* ⚡ **GPU Acceleration Option** – Faster inference for large models.
* 📊 **Source Highlighting** – Show exact PDF passages used in answers.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork this repo.
2. Create a branch for your feature.
3. Submit a pull request with a clear explanation of your changes.

---

## 📜 License

This project is licensed under the MIT License.