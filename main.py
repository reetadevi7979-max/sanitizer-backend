import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="AI Watermark Sanitizer API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SanitizeRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"status": "online", "message": "API is operational."}

@app.post("/sanitize")
async def sanitize_text(data: SanitizeRequest):
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY environment variable is missing on Render."
        )

    try:
        groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key.strip()
        )

        prompt = (
            "You are an expert editor specializing in removing AI watermarks, repetitive phrasing, "
            "and telltale structural patterns (such as unnatural transitions, passive overuse, and buzzwords). "
            "Rewrite the following text so it sounds completely human, polished, and natural while maintaining "
            "the original meaning:\n\n"
            f"{data.text}"
        )

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You sanitize text to sound strictly human and remove AI watermarks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        return {
            "status": "success", 
            "sanitized_text": response.choices[0].message.content
        }

    except Exception as e:
        # Passes the exact Groq error back to RapidAPI response
        raise HTTPException(
            status_code=500, 
            detail=f"GROQ FAILURE: {type(e).__name__} - {str(e)}"
        )
