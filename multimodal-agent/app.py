# app.py
import json
import streamlit as st
from agent.agent import run_from_bytes

# ── Page config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal Agent",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Multimodal Agent")
st.caption("Upload any image. Ask anything. Get structured output.")

# ── Layout: two equal columns ──────────────────────────────────────
# Left: inputs. Right: output.
# The user can see both panels simultaneously.

left, right = st.columns([1, 1], gap="large")

# ── Left column: inputs ────────────────────────────────────────────

with left:
    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Receipts, whiteboards, screenshots, product photos — anything works.",
    )

    # Show a preview as soon as the image is uploaded
    if uploaded_file:
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    instruction = st.text_area(
        "What do you want to know?",
        placeholder="e.g. Am I overspending? / Summarize this meeting / Should I buy this?",
        height=100,
    )

    context = st.text_area(
        "Extra context (optional)",
        placeholder="e.g. My monthly budget is Rs.10,000 / I am packing for a 3-day trip",
        height=80,
    )

    # The button is greyed out until both image and instruction are provided.
    # This prevents confusing empty-state errors.
    run_button = st.button(
        "Run Agent",
        type="primary",
        disabled=not (uploaded_file and instruction.strip()),
        use_container_width=True,
    )

# ── Right column: output ───────────────────────────────────────────

with right:
    st.subheader("Output")

    if run_button and uploaded_file and instruction.strip():

        with st.spinner("Llama 4 Scout is thinking..."):
            try:
                # uploaded_file.getvalue() returns the raw bytes of the uploaded file.
                # This is the correct Streamlit pattern — not .read(), which can only
                # be called once before the cursor moves past the end of the file.
                image_bytes = uploaded_file.getvalue()

                result = run_from_bytes(
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                    instruction=instruction,
                    context=context,
                )

                # Check if the agent returned an error dict
                if "error" in result:
                    st.error(f"Agent error: {result['error']}")
                    if "raw_output" in result:
                        st.code(result["raw_output"], language="text")
                else:
                    st.success("Done!")

                    # st.json renders a Python dict as an interactive,
                    # collapsible JSON tree in the browser
                    st.json(result)

                    # Also offer a flat copy-paste version
                    with st.expander("Raw JSON (copy-paste ready)"):
                        st.code(json.dumps(result, indent=2), language="json")

            except ValueError as e:
                # Catches unsupported image format errors from image.py
                st.error(str(e))

            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

    elif not uploaded_file:
        st.info("Upload an image to get started.")

    elif not instruction.strip():
        st.info("Type a question or instruction above.")