# agent/client.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Create the client once here.
# Every other module imports this single instance.
# This avoids creating a new connection on every function call.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))