# AI Meeting Assistant (Live Zoom & Teams Integration + Transcription + GPT Q&A + Auth + UI + OAuth + Docker)
# Tech Stack: FastAPI + OpenAI Whisper + OpenAI GPT-4 + FAISS + Zoom SDK + Microsoft Teams Bot + Azure Speech + Streamlit + OAuth2 + Docker

from fastapi import FastAPI, UploadFile, File, Request, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List
import tempfile
import openai
import whisper
import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import aiohttp

app = FastAPI()

# OAuth2 Login (simplified for demo)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load models
transcriber = whisper.load_model("base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)  # 384 dims for MiniLM
chunks = []  # Stores transcript chunks

openai.api_key = os.getenv("OPENAI_API_KEY")
API_TOKEN = os.getenv("WEBHOOK_AUTH_TOKEN")  # Shared token for webhook auth
USERNAME = os.getenv("APP_USERNAME", "admin")
PASSWORD = os.getenv("APP_PASSWORD", "secret")

class Question(BaseModel):
    question: str

# --------------------------- Basic UI (Streamlit-style) with Login ----------------------------
@app.get("/", response_class=HTMLResponse)
async def root(token: str = Depends(oauth2_scheme)):
    return """
    <html>
    <body>
        <h2>Ask Meeting Assistant</h2>
        <form action="/ask/" method="post">
            <input type="text" name="question" style="width:400px">
            <input type="submit" value="Ask">
        </form>
    </body>
    </html>
    """

@app.post("/ask/", response_class=HTMLResponse)
async def ask_web_ui(request: Request, token: str = Depends(oauth2_scheme)):
    form = await request.form()
    question = form.get("question")
    query_vec = embedder.encode([question])
    D, I = index.search(np.array(query_vec), k=3)
    relevant_chunks = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""
    Based on the following meeting transcript:
    {relevant_chunks}

    Answer this question:
    {question}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response['choices'][0]['message']['content']
    return HTMLResponse(content=f"<p><strong>Answer:</strong> {answer}</p><br><a href='/'>Ask another</a>", status_code=200)

# --------------------------- OAuth2 Token Endpoint ----------------------------
from fastapi import Form
@app.post("/token")
async def login(form_data: Request):
    data = await form_data.form()
    username = data.get("username")
    password = data.get("password")
    if username == USERNAME and password == PASSWORD:
        return {"access_token": "demo-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# --------------------------- File Upload Transcription ----------------------------
@app.post("/transcribe/")
async def transcribe_meeting(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp.flush()
        result = transcriber.transcribe(temp.name)
        transcript = result['text']

    words = transcript.split()
    chunk_size = 50
    text_chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    global chunks
    chunks = text_chunks
    vectors = embedder.encode(text_chunks)
    index.add(np.array(vectors))

    return {"message": "Transcription and indexing complete.", "chunks": len(text_chunks)}

# --------------------------- Zoom Webhook Auth & Ingestion ------------------------
@app.post("/zoom/recording_webhook")
async def zoom_recording_webhook(request: Request, authorization: str = Header(None)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid token")

    payload = await request.json()
    download_url = payload['payload']['object']['recording_files'][0]['download_url']
    access_token = os.getenv("ZOOM_JWT")

    async with aiohttp.ClientSession() as session:
        async with session.get(download_url, headers={"Authorization": f"Bearer {access_token}"}) as resp:
            if resp.status == 200:
                data = await resp.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(data)
                    result = transcriber.transcribe(temp_file.name)
                    transcript = result['text']

                words = transcript.split()
                chunk_size = 50
                text_chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                global chunks
                chunks = text_chunks
                vectors = embedder.encode(text_chunks)
                index.add(np.array(vectors))
                return {"message": "Zoom recording processed and indexed."}

    return {"message": "Failed to process Zoom recording."}

# -------------------------- Teams Webhook Auth & Ingestion ------------------------
@app.post("/teams/audio_webhook")
async def teams_audio_webhook(request: Request, authorization: str = Header(None)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid token")

    payload = await request.json()
    audio_url = payload.get("audioUrl")
    bearer_token = os.getenv("TEAMS_API_TOKEN")

    async with aiohttp.ClientSession() as session:
        async with session.get(audio_url, headers={"Authorization": f"Bearer {bearer_token}"}) as resp:
            if resp.status == 200:
                data = await resp.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(data)
                    result = transcriber.transcribe(temp_file.name)
                    transcript = result['text']

                words = transcript.split()
                chunk_size = 50
                text_chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                global chunks
                chunks = text_chunks
                vectors = embedder.encode(text_chunks)
                index.add(np.array(vectors))
                return {"message": "Teams audio processed and indexed."}

    return {"message": "Failed to process Teams audio."}

# -------------------------- Docker Support (via Dockerfile) ------------------------
# Add this to a separate Dockerfile in root directory:
# FROM python:3.10-slim
# WORKDIR /app
# COPY . .
# RUN pip install --no-cache-dir -r requirements.txt
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# To run locally:
# uvicorn main:app --reload
