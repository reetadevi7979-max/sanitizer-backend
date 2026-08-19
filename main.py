import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client, Client

app = FastAPI(title="AI Watermark Sanitizer API")

# Enable CORS for browser extension and checkout page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EXTENSION_SECRET = os.environ.get("EXTENSION_SECRET")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Accepts both SUPABASE_SERVICE_KEY or SUPABASE_KEY from Render
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")

# Initialize OpenAI Client for Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or "placeholder"
)

# Initialize Supabase Client
supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

# Request / Response Schemas
class SanitizeRequest(BaseModel):
    text: str

class SanitizeResponse(BaseModel):
    original_text: str
    sanitised_text: str
    status: str

class ConfirmSubscriptionRequest(BaseModel):
    subscription_id: str
    email: str

SYSTEM_PROMPT = """
You are an expert AI watermark sanitizer. 
Your goal is to disrupt AI detection patterns while preserving the original text's exact vocabulary, tone, and wording as close to identical as possible.

CONSTRAINTS:
1. DO NOT use heavy synonym replacements. Keep the original words wherever possible.
2. Make only subtle adjustments to sentence structure, clause order, or active/passive voice to break AI token chains.
3. STRICTLY PRESERVE all proper nouns, technical terms, numbers, dates, and core facts.
4. Return ONLY the sanitized output text with no intro or outro comments.
"""

@app.get("/")
def health_check():
    return {"status": "online", "message": "Sanitizer backend running."}

@app.post("/confirm-subscription")
async def confirm_subscription(payload: ConfirmSubscriptionRequest):
    """Called by subscribe.html after PayPal checkout completes."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not configured. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.")

    if not payload.subscription_id or not payload.email:
        raise HTTPException(status_code=400, detail="Missing subscription_id or email.")

    # Generate a unique license key (e.g., SAN-A1B2-C3D4)
    license_key = f"SAN-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}"

    try:
        # Save subscriber record to Supabase
        res = supabase.table("subscribers").insert({
            "email": payload.email,
            "subscription_id": payload.subscription_id,
            "license_key": license_key,
            "status": "active"
        }).execute()

        return {"status": "success", "license_key": license_key}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/sanitize", response_model=SanitizeResponse)
async def sanitize_text(
    payload: SanitizeRequest,
    x_extension_key: Optional[str] = Header(None, alias="x-extension-key"),
    x_license_key: Optional[str] = Header(None, alias="x-license-key")
):
    # 1. Extension Secret Verification
    if EXTENSION_SECRET and x_extension_key != EXTENSION_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized extension request.")

    # 2. License Key Verification against Supabase (if provided)
    if x_license_key and supabase:
        sub_check = supabase.table("subscribers")\
            .select("status")\
            .eq("license_key", x_license_key)\
            .execute()

        if not sub_check.data or sub_check.data[0].get("status") != "active":
            raise HTTPException(status_code=403, detail="Invalid or expired license key.")

    # 3. Input Validation
    if not payload.text or len(payload.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # 4. Groq Model Execution
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
