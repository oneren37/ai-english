from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import httpx
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()

TRANSCRIBE_API_KEY = os.getenv("TRANSCRIBE_API_KEY")


# Модуль транскрибации через Fireworks API
async def transcribe_audio(file: UploadFile) -> str:
    url = "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {TRANSCRIBE_API_KEY}"}
    data = {
        "model": "whisper-v3-turbo",
        "temperature": "0",
        "vad_model": "silero"
    }
    # Читаем содержимое файла
    file_bytes = await file.read()
    files = {"file": (file.filename, file_bytes, file.content_type)}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, files=files)
    if response.status_code == 200:
        result = response.json()
        # Предполагаем, что текст транскрипции в result["text"]
        return result.get("text", "")
    else:
        raise Exception(f"Transcription error: {response.status_code} {response.text}")

async def process_text(text: str) -> dict:
    # Пример запроса к другому API с использованием ключа
    # url = "https://your-second-api.com/process"
    # headers = {"Authorization": f"Bearer {SECOND_API_KEY}"}
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(url, json={"text": text}, headers=headers)
    #     return response.json()
    return {"result": f"Обработано: {text}"}

@app.post("/process")
async def process(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if not text and not file:
        raise HTTPException(status_code=400, detail="Нужно передать либо текст, либо аудиофайл.")

    if file:
        transcript = await transcribe_audio(file)
        result = await process_text(transcript)
    else:
        if text is None:
            raise HTTPException(status_code=400, detail="Текст не должен быть пустым.")
        result = await process_text(text)

    return JSONResponse(content=result) 