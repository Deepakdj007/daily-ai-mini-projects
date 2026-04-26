import os
import streamlit as st
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()
client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

st.set_page_config(
    page_title="Indian Language Chatbot",
    page_icon="🇮🇳",
    layout="centered"
)
st.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg", width=50)
st.title("Indian Language Chatbot")
st.caption("Powered by Sarvam AI · 22 Indian languages")

if "messages" not in st.session_state:
    st.session_state.messages = []   # for display

if "history" not in st.session_state:
    st.session_state.history = []    # for Sarvam API

with st.sidebar:
    st.header("⚙️ Settings")

    target_language = st.selectbox(
        "Response Language",
        options=[
            ("Auto-detect", "auto"),
            ("Hindi — हिंदी", "hi-IN"),
            ("Tamil — தமிழ்", "ta-IN"),
            ("Telugu — తెలుగు", "te-IN"),
            ("Malayalam — മലയാളം", "ml-IN"),
            ("Kannada — ಕನ್ನಡ", "kn-IN"),
            ("Bengali — বাংলা", "bn-IN"),
            ("Marathi — मराठी", "mr-IN"),
            ("Gujarati — ગુજરાતી", "gu-IN"),
            ("Punjabi — ਪੰਜਾਬੀ", "pa-IN"),
            ("English", "en-IN"),
        ],
        format_func=lambda x: x[0]
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def get_response(user_message: str, lang_setting: tuple) -> str:
    lang_code = lang_setting[1]

    if lang_code == "auto":
        system_content = """You are a helpful AI assistant for Indian users.
Detect the language the user writes in and always respond in that same language.
Be warm, friendly, and conversational."""
    else:
        lang_name = lang_setting[0].split(" — ")[0]
        system_content = f"""You are a helpful AI assistant.
Always respond in {lang_name}, regardless of what language the user writes in.
Be warm, friendly, and conversational."""

    # Add to API history
    st.session_state.history.append({
        "role": "user",
        "content": user_message
    })

    # Note: .completions() directly — no .create()
    response = client.chat.completions(
        model="sarvam-105b",
        messages=[
            {"role": "system", "content": system_content}
        ] + st.session_state.history[-20:]
    )

    reply = response.choices[0].message.content

    st.session_state.history.append({
        "role": "assistant",
        "content": reply
    })

    return reply

if user_input := st.chat_input("किसी भी भाषा में लिखें... / Type in any language..."):

    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("सोच रहा हूँ..."):
            response = get_response(user_input, target_language)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

