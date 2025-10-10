# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from openai import OpenAI
# from PyPDF2 import PdfReader
# import numpy as np
# import faiss
# import os
# # from dotenv import load_dotenv    
# from dotenv import dotenv_values
# from fastapi.responses import JSONResponse, HTMLResponse

# # Load environment variables from .env
# # load_dotenv()

# config = dotenv_values(".env")  # loads all key-value pairs as a dictionary
# print("All values:", config)
# print("API Key:", config["OPENAI_API_KEY"])

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# client = OpenAI(api_key=config["OPENAI_API_KEY"])


# PDF_PATH = "Covision.pdf"

# # --- PDF Processing ---
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         if page.extract_text():
#             text += page.extract_text()
#     return text

# def chunk_text(text, max_length=800):
#     return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# # --- Embedding and FAISS ---
# def get_embedding(text):
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=text
#     )
#     return response.data[0].embedding

# def build_faiss_index(chunks):
#     sample_emb = get_embedding(chunks[0])
#     dim = len(sample_emb)
#     index = faiss.IndexFlatL2(dim)
#     for chunk in chunks:
#         emb = get_embedding(chunk)
#         index.add(np.array([emb], dtype='float32'))
#     return index, chunks

# pdf_text = extract_text_from_pdf(PDF_PATH)
# chunks = chunk_text(pdf_text)
# index, chunks = build_faiss_index(chunks)

# # --- Querying ---
# def search_similar_chunks(query, index, chunks, top_k=3):
#     query_emb = np.array([get_embedding(query)], dtype='float32')
#     distances, indices = index.search(query_emb, top_k)
#     return [chunks[i] for i in indices[0]]

# def ask_about_pdf(query, index, chunks):
#     context_chunks = search_similar_chunks(query, index, chunks)
#     context = "\n".join(context_chunks)
#     prompt = f"""
# You are a helpful assistant that answers ONLY using the information below.
# If the answer is not found, say "I cannot give you that information."

# Context:
# {context}

# Question: {query}
# """
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     # ✅ Updated for OpenAI 2.x
#     return response.choices[0].message["content"].strip()

# # --- API Models ---
# class ChatRequest(BaseModel):
#     message: str

# # --- Routes ---
# @app.get("/", response_class=HTMLResponse)
# async def home():
#     return "<h2>📘 Covision Chatbot (FastAPI Version)</h2><p>Use the /Chat endpoint to send messages.</p>"

# @app.post("/Chat")
# async def chat_endpoint(request: ChatRequest):
#     try:
#         user_message = request.message.strip()
#         if not user_message:
#             return JSONResponse(content={"error": "Message cannot be empty"}, status_code=400)

#         ai_response = ask_about_pdf(user_message, index, chunks)
#         return {"success": True, "user_message": user_message, "ai_response": ai_response}
#     except Exception as e:
#         print("❌ Backend error:", e)
#         return JSONResponse(content={"error": f"Error: {str(e)}"}, status_code=500)

# # --- Run locally ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5000)






from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import faiss
import os
from dotenv import load_dotenv, dotenv_values
from fastapi.responses import JSONResponse, HTMLResponse

# -------------------------------
# STEP 1: Load environment variables (Works both locally & on Railway)
# -------------------------------
if os.getenv("RAILWAY_ENVIRONMENT") is None:
    # Running locally
    load_dotenv()
    print("📦 Running locally: Loaded .env file.")
    config = dotenv_values(".env")
    api_key = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
else:
    # Running on Railway
    print("🚀 Running on Railway: Using environment variables.")
    api_key = os.getenv("OPENAI_API_KEY")

# Debugging output (safe)
if api_key:
    print("✅ API Key loaded (starts with):", api_key[:10], "...")
    print("🔒 Key length:", len(api_key))
else:
    print("❌ ERROR: OPENAI_API_KEY not found in environment!")
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# -------------------------------
# STEP 2: Initialize FastAPI and OpenAI Client
# -------------------------------
app = FastAPI()
client = OpenAI(api_key=api_key)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# STEP 3: PDF Handling
# -------------------------------
PDF_PATH = "Covision.pdf"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def chunk_text(text, max_length=800):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# -------------------------------
# STEP 4: Embeddings and FAISS Index
# -------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_faiss_index(chunks):
    sample_emb = get_embedding(chunks[0])
    dim = len(sample_emb)
    index = faiss.IndexFlatL2(dim)
    for chunk in chunks:
        emb = get_embedding(chunk)
        index.add(np.array([emb], dtype='float32'))
    return index, chunks

pdf_text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(pdf_text)
index, chunks = build_faiss_index(chunks)

# -------------------------------
# STEP 5: Query Logic
# -------------------------------
def search_similar_chunks(query, index, chunks, top_k=3):
    query_emb = np.array([get_embedding(query)], dtype='float32')
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0]]

def ask_about_pdf(query, index, chunks):
    context_chunks = search_similar_chunks(query, index, chunks)
    context = "\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant that answers ONLY using the information below.
If the answer is not found, say "I cannot give you that information."

Context:
{context}

Question: {query}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"].strip()

# -------------------------------
# STEP 6: FastAPI Endpoints
# -------------------------------
class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h2>📘 Covision Chatbot (FastAPI Version)</h2><p>Use the /Chat endpoint to send messages.</p>"

@app.post("/Chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            return JSONResponse(content={"error": "Message cannot be empty"}, status_code=400)

        ai_response = ask_about_pdf(user_message, index, chunks)
        return {"success": True, "user_message": user_message, "ai_response": ai_response}
    except Exception as e:
        print("❌ Backend error:", e)
        return JSONResponse(content={"error": f"Error: {str(e)}"}, status_code=500)

# -------------------------------
# STEP 7: Local Run
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
