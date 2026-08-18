import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="AI Watermark Sanitizer API")

# Enable CORS for browser extensions and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or "placeholder"
)

class SanitizeRequest(BaseModel):
    text: str

class SanitizeResponse(BaseModel):
    original_text: str
    sanitised_text: str
    status: str
SYSTEM_PROMPT = """
You are an expert AI watermark sanitizer. 
Your goal is to disrupt AI detection patterns while preserving the original text's exact vocabulary, tone, and wording as close to identical as possible.

CONSTRAINTS:
1. DO NOT use heavy synonym replacements (e.g., do NOT change 'lazy dog' to 'sluggish canine'). Keep the original words wherever possible.
2. Make only subtle adjustments to sentence structure, clause order, or active/passive voice to break AI token chains.
3. STRICTLY PRESERVE all proper nouns, technical terms, numbers, dates, and core facts.
4. Return ONLY the sanitized output text with no intro or outro comments.
"""

@app.get("/")
def health_check():
    return {"status": "online", "message": "Sanitizer backend running."}

@app.post("/sanitize", response_model=SanitizeResponse)
async def sanitize_text(payload: SanitizeRequest):
    if not payload.text or len(payload.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.text}
            ]
        )
        
        return SanitizeResponse(
            original_text=payload.text,
            sanitised_text=response.choices[0].message.content.strip(),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
