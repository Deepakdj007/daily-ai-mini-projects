"""Streamlit UI for the real-time voice translator.

User picks a target language, clicks Start, and speaks. Gemini Live
translates speech to the target language in real time and plays it back
through the browser speaker. A live transcript shows both sides.
"""

import queue

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.audio_processor import AudioTranslatorProcessor
from src.config import GEMINI_API_KEY, LANGUAGES

# -----------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------
st.set_page_config(page_title="Voice Translator", page_icon="🌐", layout="centered")

st.title("🌐 Real-Time Voice Translator")
st.caption("Powered by Gemini Live · Speak any language, hear any language back")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is missing. Copy `.env.example` to `.env` and paste your key "
        "from https://aistudio.google.com/apikey"
    )
    st.stop()

# -----------------------------------------------------------------------
# Language selector
# -----------------------------------------------------------------------
col_lang, col_info = st.columns([1, 2])

with col_lang:
    target_name = st.selectbox("Translate to:", list(LANGUAGES.keys()), index=0)

with col_info:
    st.info(
        f"Source language is auto-detected. "
        f"Speak anything — the AI replies in **{target_name}** in real time."
    )

target_code = LANGUAGES[target_name]

# Clear transcript when the target language changes
if st.session_state.get("active_lang") != target_code:
    st.session_state.active_lang = target_code
    st.session_state.transcripts = []

# -----------------------------------------------------------------------
# WebRTC audio component
# Each language gets its own key so the component restarts on language change,
# creating a fresh AudioTranslatorProcessor bound to the new target language.
# -----------------------------------------------------------------------
ctx = webrtc_streamer(
    key=f"translator-{target_code}",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=lambda: AudioTranslatorProcessor(target_code),
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

# Store the processor in session_state so the fragment can access it
# across its own independent re-runs without re-evaluating this script.
is_active = bool(ctx and ctx.state.playing)
if is_active and ctx.audio_processor:
    st.session_state.processor = ctx.audio_processor
elif not is_active:
    st.session_state.pop("processor", None)

# -----------------------------------------------------------------------
# Live transcript — auto-refreshes every 0.5 s via @st.fragment
# -----------------------------------------------------------------------
st.divider()

header_col, btn_col = st.columns([3, 1])
with header_col:
    st.subheader("Live Transcript")
with btn_col:
    if st.button("Clear", use_container_width=True):
        st.session_state.transcripts = []


@st.fragment(run_every=0.5)
def show_transcripts() -> None:
    """Drain new transcript items from the processor queue every 0.5 s."""
    processor: AudioTranslatorProcessor | None = st.session_state.get("processor")

    if processor is not None:
        while True:
            try:
                item = processor.transcript_queue.get_nowait()
                st.session_state.transcripts.append(item)
            except queue.Empty:
                break

    transcripts: list[dict] = st.session_state.get("transcripts", [])

    # Status line
    if not transcripts:
        if is_active:
            st.caption("🎙️ Listening... start speaking.")
        else:
            st.caption("Click **START** above, allow mic access, then speak.")
        return

    # Chat-style display — You on right (user), translation on left (assistant)
    for item in transcripts[-40:]:
        if item["type"] == "error":
            st.error(f"Session error: {item['text']}")
        elif item["type"] == "input":
            with st.chat_message("user"):
                st.write(item["text"])
        else:
            with st.chat_message("assistant"):
                st.write(f"**{target_name}:** {item['text']}")


show_transcripts()
