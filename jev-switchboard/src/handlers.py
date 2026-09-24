"""The handlers a route can land on: the store tool, a small LLM or a big LLM.

Inputs:  a customer message and the Route that Jev's answers produced.
Outputs: (reply text, context the reply was allowed to use, ms spent).
"""

import time

from groq import AsyncGroq

from src import config, store

groq = AsyncGroq()  # reads GROQ_API_KEY, which config.load_dotenv() has loaded

PRODUCT_PROMPT = f"""You answer product questions for Kumkum Home, an Indian home-decor store.
Use only this catalog. If the answer is not in it, say you will check with the team.
Reply in 2-3 short sentences.

CATALOG:
{store.CATALOG}"""

POLICY_PROMPT = """You handle returns and complaints for Kumkum Home, an Indian home-decor store.
Follow the policy exactly. Never offer anything the policy does not allow.
Reply in 3-4 short sentences.

POLICY:
{policy}

ORDER FACTS:
{facts}"""


def order_facts(message: str) -> str:
    """Build the order facts the LLM may use. The day count is computed here."""
    order_id = store.find_order_id(message)
    order = store.ORDERS.get(order_id or "")
    if order is None:
        return "No order ID given."
    days = store.days_since_delivery(order_id)
    if days is None:
        return f"{order_id}: {order['item']}, Rs {order['price']}, not delivered yet."
    window = "inside" if days <= store.RETURN_WINDOW_DAYS else "outside"
    return (f"{order_id}: {order['item']}, Rs {order['price']}, delivered {days} days ago, "
            f"{window} the {store.RETURN_WINDOW_DAYS}-day return window.")


async def ask_llm(model: str, system: str, message: str) -> str:
    """One chat call to Groq with a system prompt."""
    response = await groq.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


async def handle(route_name: str, message: str) -> tuple[str, str, float]:
    """Run the handler for a route. Returns (reply, context the reply may use, ms)."""
    start = time.perf_counter()
    if route_name == "clarify":
        reply, context = "Could you tell me a bit more? An order ID helps too.", ""
    elif route_name == "human":
        reply, context = "I'm passing you to a teammate now. They'll reply within 2 hours.", ""
    elif route_name == "tool":
        reply, context = store.order_lookup(message), ""
    elif route_name == "small_llm":
        context = store.CATALOG
        reply = await ask_llm(config.SMALL_LLM, PRODUCT_PROMPT, message)
    else:
        facts = order_facts(message)
        context = f"{store.RETURN_POLICY}\n{facts}"
        prompt = POLICY_PROMPT.format(policy=store.RETURN_POLICY, facts=facts)
        reply = await ask_llm(config.BIG_LLM, prompt, message)
    return reply, context, (time.perf_counter() - start) * 1000
