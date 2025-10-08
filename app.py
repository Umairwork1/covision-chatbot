from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, HTMLResponse
app = FastAPI()

load_dotenv()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
print("🔍 Debug: Loaded API key value =", api_key)  # Shows the raw key
if not api_key:
    print("❌ API key not found! Make sure your .env file exists and has OPENAI_API_KEY set.")
else:
    print("✅ API key loaded successfully (starts with):", api_key)
client = OpenAI(api_key=api_key)


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

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def build_faiss_index(chunks):
    sample_emb = get_embedding(chunks[0])
    dim = len(sample_emb)
    index = faiss.IndexFlatL2(dim)
    for chunk in chunks:
        emb = get_embedding(chunk)
        index.add(np.array([emb]).astype('float32'))
    return index, chunks

pdf_text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(pdf_text)
index, chunks = build_faiss_index(chunks)

def search_similar_chunks(query, index, chunks, top_k=3):
    query_emb = np.array([get_embedding(query)]).astype('float32')
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
    return response.choices[0].message.content.strip()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
