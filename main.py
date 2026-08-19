import os
import random
import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import resend
from openai import OpenAI
from supabase import create_client, Client

app = FastAPI(title="AI Watermark Sanitizer API")

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
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or "placeholder"
)

supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)

# Request Models
class SanitizeRequest(BaseModel):
    text: str

class ConfirmSubscriptionRequest(BaseModel):
    subscription_id: str
    email: str

class RequestOTPRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str

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

@app.get("/subscribe", response_class=FileResponse)
async def serve_subscribe_page():
    return FileResponse("subscribe.html")

@app.post("/confirm-subscription")
async def confirm_subscription(payload: ConfirmSubscriptionRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")
    
    email = payload.email.lower().strip()
    
    # Save or update subscriber status to active
    supabase.table("subscribers").upsert({
        "email": email,
        "subscription_id": payload.subscription_id,
        "status": "active"
    }, on_conflict="email").execute()

    return {"status": "success", "message": "Subscription confirmed."}

@app.post("/auth/request-otp")
async def request_otp(payload: RequestOTPRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database error.")
    
    email = payload.email.lower().strip()
    
    # Check if subscriber is active
    res = supabase.table("subscribers").select("status").eq("email", email).execute()
    if not res.data or res.data[0].get("status") != "active":
        raise HTTPException(status_code=403, detail="No active subscription found for this email.")

    # Generate 6-digit OTP code
    otp_code = f"{random.randint(100000, 999999)}"
    
    # Store OTP in Supabase with expiration (10 minutes)
    supabase.table("subscribers").update({
        "otp_code": otp_code,
        "otp_created_at": datetime.datetime.utcnow().isoformat()
    }).eq("email", email).execute()

    # Send OTP Email
    if RESEND_API_KEY:
        resend.Emails.send({
            "from": "AI Sanitizer <onboarding@resend.dev>",
            "to": email,
            "subject": "Your AI Sanitizer Login Code",
            "html": f"""
                <h2>AI Sanitizer Login</h2>
                <p>Your 6-digit verification code is:</p>
                <h1 style="color:#2563eb; letter-spacing: 4px; font-size:32px;">{otp_code}</h1>
                <p>This code will expire in 10 minutes.</p>
            """
        })

    return {"status": "success", "message": "Verification code sent to your email."}

@app.post("/auth/verify-otp")
async def verify_otp(payload: VerifyOTPRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database error.")
    
    email = payload.email.lower().strip()
    
    res = supabase.table("subscribers").select("otp_code, status").eq("email", email).execute()
    if not res.data or res.data[0].get("status") != "active":
        raise HTTPException(status_code=403, detail="No active subscription.")

    stored_otp = res.data[0].get("otp_code")
    if not stored_otp or stored_otp != payload.otp.strip():
        raise HTTPException(status_code=400, detail="Invalid verification code.")

    # Clear OTP after successful login
    supabase.table("subscribers").update({"otp_code": None}).eq("email", email).execute()

    return {"status": "success", "email": email}

@app.post("/sanitize")
async def sanitize_text(
    payload: SanitizeRequest,
    x_extension_key: Optional[str] = Header(None, alias="x-extension-key"),
    x_user_email: Optional[str] = Header(None, alias="x-user-email")
):
    if EXTENSION_SECRET and x_extension_key != EXTENSION_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized extension request.")

    if not x_user_email or not supabase:
        raise HTTPException(status_code=401, detail="Please sign in to use AI Sanitizer.")

    # Verify subscription status on every request
    res = supabase.table("subscribers").select("status").eq("email", x_user_email.lower().strip()).execute()
    if not res.data or res.data[0].get("status") != "active":
        raise HTTPException(status_code=403, detail="Active subscription required.")

    if not payload.text or len(payload.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.text}
            ]
        )
        return {
            "original_text": payload.text,
            "sanitised_text": response.choices[0].message.content.strip(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
