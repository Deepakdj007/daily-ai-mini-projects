# agent/prompt.py

# This system prompt is the key to dynamic output.
# Notice what it does NOT say:
# it does NOT list fields for the model to fill in.
# It tells the model to DECIDE the right structure itself.
SYSTEM_PROMPT = """You are a multimodal reasoning agent.

You will receive an image and a question or instruction about it.
You may also receive additional context from the user.

Your job:
1. Look at the image carefully
2. Understand what the user is asking
3. Decide what information is most useful to extract
4. Decide what JSON structure best represents your answer for THIS specific image and question

Rules:
- Respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks.
- Choose field names that are clear and self-explanatory
- Use arrays for lists of items
- Use nested objects when grouping related fields makes sense
- All numeric values must be plain numbers, not strings with units
- Be specific and actionable — vague answers are not useful
"""


def build_messages(
    base64_image: str,
    media_type: str,
    instruction: str,
    context: str = "",
) -> list[dict]:
    """
    Assemble the multimodal message list that gets sent to Llama 4 Scout.

    The user message content is a LIST — this is how multimodal requests work.
    Text and image are separate items in the same list.
    The model receives both and reasons across them jointly.
    """

    # Combine the user's instruction with any extra context they provided
    user_text = instruction
    if context.strip():
        user_text += f"\n\nAdditional context from user:\n{context.strip()}"

    return [
        # System message: sets the agent's behavior and output rules
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        # User message: contains BOTH the text instruction AND the image
        # This is what makes the request multimodal
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # data URI: embeds the image bytes directly in the request
                        # format: data:{media_type};base64,{encoded_bytes}
                        "url": f"data:{media_type};base64,{base64_image}"
                    },
                },
            ],
        },
    ]