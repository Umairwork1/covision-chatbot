from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from flask_cors import CORS
from PyPDF2 import PdfReader
import numpy as np
import faiss
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

CORS(app) 

# ✅ Add your API key

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

print("📘 Loading and indexing PDF...")
pdf_text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(pdf_text)
index, chunks = build_faiss_index(chunks)
print("✅ PDF loaded and indexed!")


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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Chat', methods=['POST', 'OPTIONS'])
def chat_endpoint():
    if request.method == 'OPTIONS':
        # Handle browser preflight
        return jsonify({'message': 'CORS preflight successful'}), 200
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" parameter'}), 400
        user_message = data['message']
        if not user_message.strip():
            return jsonify({'error': 'Message cannot be empty'}), 400
        ai_response = ask_about_pdf(user_message, index, chunks)
        return jsonify({'success': True, 'user_message': user_message, 'ai_response': ai_response})
    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({'error': f'Error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
