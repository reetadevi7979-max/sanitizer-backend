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

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Initialize Resend & Groq
resend.api_key = os.getenv("RESEND_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = None
if GROQ_API_KEY:
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )

# In-memory store for OTPs
otp_store = {}


# --- Request Models ---

class OTPRequest(BaseModel):
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str

class SanitizeRequest(BaseModel):
    text: str

class SubscriptionRequest(BaseModel):
    email: str
    subscription_id: Optional[str] = None


# --- Static Routes ---

@app.get("/subscribe")
async def serve_subscribe_page():
    file_path = os.path.join(os.path.dirname(__file__), "subscribe.html")
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, 
            detail="subscribe.html not found in project root directory."
        )
    return FileResponse(file_path, media_type="text/html")


# --- Auth Routes ---

@app.post("/send-otp")
async def send_otp(data: OTPRequest):
    email = data.email.lower().strip()

    if not supabase:
        raise HTTPException(status_code=500, detail="Database client not configured.")

    try:
        response = supabase.table("subscribers").select("*").eq("email", email).execute()
        if not response.data or response.data[0].get("status") != "active":
            raise HTTPException(
                status_code=403, 
                detail="No active subscription found for this email."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    otp = str(random.randint(100000, 999999))
    otp_store[email] = {
        "otp": otp,
        "expires_at": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)
    }

    try:
        resend.Emails.send({
            "from": "AI Sanitizer <onboarding@resend.dev>",
            "to": email,
            "subject": "Your AI Sanitizer Verification Code",
            "html": f"<p>Your 6-digit login code is: <strong>{otp}</strong>. It expires in 10 minutes.</p>"
        })
        return {"status": "success", "message": "OTP sent successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


@app.post("/verify-otp")
async def verify_otp(data: VerifyOTPRequest):
    email = data.email.lower().strip()
    user_otp = data.otp.strip()

    record = otp_store.get(email)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP code requested for this email.")

    if datetime.datetime.now(datetime.timezone.utc) > record["expires_at"]:
        del otp_store[email]
        raise HTTPException(status_code=400, detail="OTP code has expired.")

    if record["otp"] != user_otp:
        raise HTTPException(status_code=400, detail="Invalid OTP code.")

    del otp_store[email]
    return {"status": "success", "message": "Authentication successful.", "email": email}


# --- Payment Routes ---

@app.post("/confirm-subscription")
async def confirm_subscription(data: SubscriptionRequest):
    email = data.email.lower().strip()
    subscription_id = data.subscription_id

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured.")

    try:
        payload = {
            "email": email,
            "status": "active"
        }
        if subscription_id:
            payload["subscription_id"] = subscription_id

        supabase.table("subscribers").upsert(
            payload, 
            on_conflict="email"
        ).execute()

        return {"status": "success", "message": "Subscription confirmed!"}

    except Exception as e:
        print(f"Supabase Insert Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database execution error: {str(e)}"
        )


# --- Core Sanitization Route ---

@app.post("/sanitize")
async def sanitize_text(data: SanitizeRequest, x_user_email: Optional[str] = Header(None)):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API not configured on server.")

    # Only validate Supabase subscription if an email header is explicitly passed (e.g., from extension)
    if x_user_email and supabase:
        res = supabase.table("subscribers").select("status").eq("email", x_user_email.lower().strip()).execute()
        if not res.data or res.data[0].get("status") != "active":
            raise HTTPException(status_code=403, detail="Active subscription required.")

    prompt = (
        "You are an expert editor specializing in removing AI watermarks, repetitive phrasing, "
        "and telltale structural patterns (such as unnatural transitions, passive overuse, and buzzwords). "
        "Rewrite the following text so it sounds completely human, polished, and natural while maintaining "
        "the original meaning:\n\n"
        f"{data.text}"
    )

    try:
       # Change this line inside @app.post("/sanitize")
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",  # Updated to active Groq production model
    messages=[
        {"role": "system", "content": "You sanitize text to sound strictly human and remove AI watermarks."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
)
        raise HTTPException(status_code=500, detail=f"Sanitization error: {str(e)}")
