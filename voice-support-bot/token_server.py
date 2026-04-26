import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

load_dotenv(".env.local")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your own domain in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/get-token")
async def get_token(user_id: str = "guest"):
    """
    Generate a short-lived LiveKit JWT for the frontend caller.
    The frontend calls this endpoint first, then uses the token
    to connect to the LiveKit room.
    """
    token = (
        api.AccessToken(
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room="support-room",
            )
        )
        .with_identity(user_id)
        .to_jwt()
    )

    return {
        "token": token,
        "livekit_url": os.environ["LIVEKIT_URL"],
    }