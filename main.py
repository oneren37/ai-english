from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import httpx
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")


# Модуль транскрибации через Fireworks API
async def transcribe_audio(file: UploadFile) -> str:
    url = "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}
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

# Обработка текста через LLM Fireworks
async def process_text(text: str) -> dict:
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIREWORKS_API_KEY}"
    }
    system_prompt = (
        "You are an English teacher. Your task is to help the user improve their English text.\n\n"
        "1. First, show the corrected version of the user's text. Highlight each correction by striking through the original mistake and showing the correction next to it, using this format: ~~mistake~~ → correction. Use MarkdownV2 formatting for Telegram.\n\n"
        "2. Then, in a separate block, briefly explain the main mistakes. The explanation must be concise and not longer than the user's original message. If the user's message is short, your explanation should be even shorter. Do not write long explanations or multiple paragraphs.\n\n"
        "Example output:\n"
        "*Corrected text:*\n"
        "~~I has~~ → I have a dog\\. ~~He like~~ → He likes to play\\.\n\n"
        "*Explanation:*\n"
        "Verb agreement: \"has\" → \"have\", \"like\" → \"likes\"\\.\n\n"
        "Now, process the following text:"
    )
    payload = {
        "model": "accounts/fireworks/models/qwen3-30b-a3b",
        "max_tokens": 5000,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return {"result": content}
        elif response.status_code == 403:
            raise HTTPException(status_code=403, detail="Unauthorized: проверьте API-ключ для LLM")
        else:
            raise HTTPException(status_code=500, detail=f"LLM error: {response.status_code} {response.text}")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="LLM API timeout (превышено время ожидания ответа)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM unknown error: {str(e)}")

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