import os
import re
import cv2
import pytesseract
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Helpful answer:
"""

# ---------- OCR IMAGE HANDLER ----------
def extract_text_from_image(image_path: str) -> str:
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
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

def load_groq_llm():
    # Increase max_tokens to allow longer summaries
    return ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1500
    )

# ---------- UTILITIES ----------
def looks_like_boilerplate(text: str) -> bool:
    # Simple heuristics to detect copyright/boilerplate pages
    return bool(re.search(r'copyright|all rights reserved|gale group|publisher|isbn|doi', text, re.I))

# ---------- BOT CREATION ----------
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

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
    cl.user_session.set("groq", load_groq_llm())

    files = await cl.AskFileMessage(
        content="👋 Hi! Upload **PDFs or Images** — I’ll explain them with Groq by default.\n\nYou can also:\n- `local: question` → search uploaded PDFs (FAISS + Local Llama)\n- `groq: question` → ask Groq directly.",
        accept=["application/pdf", "image/png", "image/jpeg"],
        max_size_mb=50,
        max_files=5
    ).send()

    if files:
        groq_llm = cl.user_session.get("groq")
        chain = cl.user_session.get("chain")
        db = chain.retriever.vectorstore

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

        for file in files:
            file_path = f"./.files/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.content)

            ext = os.path.splitext(file_path)[-1].lower()
            raw_docs = []
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()  # list of Documents (per-page)
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                txt = extract_text_from_image(file_path)
                raw_docs = [Document(page_content=txt, metadata={"source": file.name})]

            if not raw_docs:
                continue

            # 1) Split into smaller chunks (preserve metadata)
            chunks = text_splitter.split_documents(raw_docs)

            # 2) Save chunks into FAISS with metadata
            texts = [c.page_content for c in chunks]
            metadatas = []
            for c in chunks:
                md = dict(c.metadata) if c.metadata else {}
                md.update({"source": file.name})
                metadatas.append(md)
            if texts:
                db.add_texts(texts, metadatas=metadatas)
                db.save_local(DB_FAISS_PATH)

            # 3) Summarize: skip obvious boilerplate first, summarize important chunks, then synthesize
            chunk_summaries = []
            # summarization prompt for each chunk
            for i, c in enumerate(chunks):
                # skip pure boilerplate but keep some if doc is small
                if looks_like_boilerplate(c.page_content) and len(chunks) > 4:
                    continue

                chunk_prompt = f"""You are a knowledgeable assistant. This is chunk {i+1} of {len(chunks)} from the document '{file.name}'.
Metadata: {c.metadata}

==== CHUNK START ====
{c.page_content}
==== CHUNK END ====

Task:
- Produce a detailed, structured summary of this chunk.
- Include important facts, numbers, definitions, and anything a researcher should know.
- If there are headings or lists, call them out.
Return only the summary text (no commentary about not knowing)."""
                res = await groq_llm.ainvoke(chunk_prompt)
                chunk_summaries.append(res.content.strip())

            if not chunk_summaries:
                # Fallback: summarize whole text (if everything was considered boilerplate)
                full_text = "\n\n".join(texts[:10])  # limit to first 10 chunks to avoid token overflow
                res = await groq_llm.ainvoke(
                    f"""You are a knowledgeable assistant.
Here is a document excerpt (possibly large). Summarize in detail and structure the summary into sections. Exclude standard copyright/legal boilerplate unless important.
{full_text}
"""
                )
                final_summary = res.content
            else:
                # 4) Synthesize chunk summaries into one long detailed summary
                synth_prompt = f"""You are a senior researcher assistant. Below are detailed summaries for chunks of a document named '{file.name}'.
Combine and synthesize them into a single, comprehensive, well-structured and long summary of the entire document.
Preserve key facts, organize into sections, create a short 'Key Takeaways' list at the end, and include page references where possible.

Chunk summaries:
{'\n\n---\n\n'.join(chunk_summaries)}
"""
                synth_res = await groq_llm.ainvoke(synth_prompt)
                final_summary = synth_res.content

            await cl.Message(content=final_summary + f"\n\n(Processed from {file.name} by Groq)").send()

    await cl.Message(content="📌 Ready! Ask your questions anytime.").send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    groq_llm = cl.user_session.get("groq")

    if chain is None or groq_llm is None:
        await message.reply("Bot not initialized, please restart.")
        return

    user_input = message.content.strip()

    # --- explicit Groq ---
    if user_input.lower().startswith("groq:"):
        query = user_input.replace("groq:", "").strip()
        res = await groq_llm.ainvoke(query)
        await cl.Message(content=res.content + "\n\n(Answered by Groq API)").send()
        return

    # --- explicit Local ---
    if user_input.lower().startswith("local:"):
        query = user_input.replace("local:", "").strip()
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        res = await chain.ainvoke(query, callbacks=[cb])
        answer = res["result"]
        sources = res["source_documents"]

        if sources:
            srcs = [f"{doc.metadata.get('source','Unknown')} (page {doc.metadata.get('page','N/A')})" for doc in sources]
            answer += "\n\nSources:\n" + "\n".join(srcs)

        await cl.Message(content=answer).send()
        return

    # --- default: detect PDF context in FAISS and use Groq with top-k chunks ---
    db = chain.retriever.vectorstore
    retriever = db.as_retriever(search_kwargs={'k': 5})
    docs = retriever.get_relevant_documents(user_input)

    # If FAISS returns meaningful document chunks, use Groq with those as context
    non_empty_docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if non_empty_docs:
        # combine top-k but keep within token limits
        context_parts = []
        for d in non_empty_docs[:5]:
            snippet = d.page_content.strip()
            # shorten overly long snippets
            if len(snippet) > 2000:
                snippet = snippet[:2000] + "..."
            context_parts.append(f"Source: {d.metadata.get('source','Unknown')} (page {d.metadata.get('page','N/A')})\n{snippet}")

        context = "\n\n---\n\n".join(context_parts)
        groq_prompt = f"""You are a helpful assistant. Use the following context (from uploaded PDFs) to answer the user's question.
Context excerpts:
{context}

User question:
{user_input}

Please answer in a detailed, well-structured manner and cite the source names/pages when relevant.
"""
        groq_res = await groq_llm.ainvoke(groq_prompt)
        answer = groq_res.content + "\n\n(Sourced from your PDFs via Groq)"
    else:
        # Fall back to Groq general
        groq_res = await groq_llm.ainvoke(user_input)
        answer = groq_res.content + "\n\n(No PDF context — answered by Groq API)"

    await cl.Message(content=answer).send()
