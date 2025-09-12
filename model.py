import os
import cv2
import pytesseract
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Do not include source references or metadata.

Helpful answer:
"""

# ---------- OCR IMAGE HANDLER ----------
def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image using OpenCV + pytesseract"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng")
    return text.strip()

# ---------- PROMPT ----------
def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

# ---------- RETRIEVAL CHAIN ----------
def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 1}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# ---------- LLM LOADERS ----------
def load_llm():
    """Local Llama2 model"""
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

def load_groq_llm():
    """Groq API model"""
    return ChatGroq(
        temperature=0.5,
        model_name="llama3-70b-8192",   # ✅ updated model
        api_key=os.getenv("GROQ_API_KEY")
    )

# ---------- BOT CREATION ----------
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # ensure FAISS index exists
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH, exist_ok=True)
        db = FAISS.from_texts(["Medical bot initialized."], embeddings)
        db.save_local(DB_FAISS_PATH)

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)

# ---------- CHAINLIT EVENTS ----------
@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)

    # ask user to upload PDFs or Images
    files = await cl.AskFileMessage(
        content="👋 Hi! Welcome to Medical Bot.\n\nUpload **PDFs or Images** to extend my knowledge base.",
        accept=["application/pdf", "image/png", "image/jpeg"],
        max_size_mb=20,
        max_files=3
    ).send()

    if files:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        for file in files:
            file_path = f"./.files/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.content)

            ext = os.path.splitext(file_path)[-1].lower()
            text = ""
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text = "\n".join([d.page_content for d in docs])
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                text = extract_text_from_image(file_path)

            if text:
                db.add_texts([text])

        db.save_local(DB_FAISS_PATH)
        await cl.Message(content=f"✅ Uploaded {len(files)} file(s) and updated knowledge base.").send()

    await cl.Message(content="📌 You can now ask your questions.\n- Use `local: question`\n- Or `groq: question`").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        await message.reply("Chain not initialized, please restart the bot.")
        return

    user_input = message.content.strip()

    # --- explicit Groq choice ---
    if user_input.lower().startswith("groq:"):
        query = user_input.replace("groq:", "").strip()
        groq_llm = load_groq_llm()
        res = await groq_llm.ainvoke(query)
        await cl.Message(content=res.content + "\n\n(Answered by Groq API)").send()
        return

    # --- explicit Local choice ---
    if user_input.lower().startswith("local:"):
        query = user_input.replace("local:", "").strip()
    else:
        query = user_input  # default → try local first

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await chain.ainvoke(query, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        source_texts = []
        for doc in sources:
            meta = doc.metadata
            src = meta.get('source', 'Unknown source')
            page = meta.get('page', 'N/A')
            source_texts.append(f"{src} (page {page})")
        answer += "\n\nSources:\n" + "\n".join(source_texts)
    else:
        # fallback → Groq if no local match
        groq_llm = load_groq_llm()
        groq_res = await groq_llm.ainvoke(query)
        answer = groq_res.content + "\n\n(No matching sources in uploaded PDFs/Images — answered by Groq API)"

    await cl.Message(content=answer).send()
