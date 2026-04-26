import os
from dotenv import load_dotenv
from sarvamai import SarvamAI


load_dotenv()

MAX_HISTORY = 20  # last 20 messages = 10 turns
LANGUAGE_NAMES = {
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ml-IN": "Malayalam",
    "kn-IN": "Kannada",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
    "en-IN": "English",
}


client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
conversation_history = []


def detect_language(text: str) -> str:
    """Returns the BCP-47 language code of the input text."""
    response = client.text.identify_language(input=text)
    return response.language_code


def chat(user_message: str) -> tuple[str | None, str]:

    lang_code = detect_language(user_message)
    lang_name = LANGUAGE_NAMES.get(lang_code, "Hindi")

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message})

    # Tell the model exactly which language to use
    dynamic_system = {
        "role": "system",
        "content": f"""You are a helpful assistant.
        The user is writing in {lang_name}. Always respond in {lang_name}.
        Be conversational, warm, and concise.""",
    }

    response = client.chat.completions(
        model="sarvam-105b", messages=[dynamic_system] + conversation_history[-20:]
    )

    reply = response.choices[0].message.content

    # Save reply to history
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, lang_code


def translate_text(
    text: str, target_language: str, source_language: str = "auto"
) -> str:
    response = client.text.translate(
        input="Your payment is pending",
        source_language_code="en-IN",
        target_language_code="hi-IN",
        model="mayura:v1",
        mode="modern-colloquial",
        # → आपका payment pending है
    )
    return response.translated_text


print(translate_text("आज मौसम बहुत अच्छा है", target_language="ta-IN"))
