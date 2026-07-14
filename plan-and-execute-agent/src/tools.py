"""Keyless tools the agents share: Wikipedia search/read and a safe calculator.

Every tool is async, returns a short string, and never raises — on failure it
returns an "error: ..." string so the agent can read it and recover. That keeps
the ReAct loop simple: a tool result is always text.
"""

import ast
import asyncio
import operator as op

import httpx
from langchain_core.tools import BaseTool, tool

# Wikipedia asks every API client to send a descriptive User-Agent. Without one
# the REST endpoint can return 403.
_HEADERS = {"User-Agent": "plan-and-execute-agent/0.1 (datasciencebrain teaching demo)"}
_TIMEOUT = 15.0
_WIKI_API = "https://en.wikipedia.org/w/api.php"


async def _wiki_get(url: str, params: dict | None = None) -> httpx.Response:
    """GET a Wikipedia URL, retrying on 429/5xx with short backoff.

    Unauthenticated Wikipedia endpoints rate-limit bursts with 429. Retrying inside
    the tool means a transient limit self-heals without wasting an agent iteration.
    """
    delay = 1.0
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as client:
        for attempt in range(3):
            resp = await client.get(url, params=params)
            if resp.status_code in (429, 502, 503) and attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
    resp.raise_for_status()
    return resp


@tool
async def search_wikipedia(query: str) -> str:
    """Search Wikipedia for articles matching a natural-language query.

    Returns up to 5 article titles. Handles plain questions like 'tallest mountain
    in Japan' — pass a normal phrase, not quotes or exact titles.
    """
    # Full-text search (list=search) matches natural phrasing, unlike opensearch's
    # prefix match — so 'tallest mountain in Japan' resolves instead of returning
    # nothing and sending the agent into a query-reformulation loop.
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 5, "format": "json",
    }
    try:
        resp = await _wiki_get(_WIKI_API, params=params)
        hits = resp.json()["query"]["search"]
    except Exception as exc:  # noqa: BLE001 — hand the failure back to the agent
        return f"error: wikipedia search failed ({exc})"
    if not hits:
        return f"no Wikipedia articles found for '{query}'"
    return "Found articles: " + "; ".join(hit["title"] for hit in hits)


@tool
async def read_wikipedia(title: str) -> str:
    """Read the full intro section of a Wikipedia article by its exact title.

    Returns the lead section as plain text. Use this to pull specific facts like a
    height, elevation, population, or date after finding the title with search.
    """
    # prop=extracts with exintro returns the WHOLE lead section (several paragraphs)
    # as plaintext — not the one-sentence REST summary, which often omits the very
    # number you need (e.g. the Eiffel Tower's height). redirects follows "Fuji" ->
    # "Mount Fuji" and the like.
    params = {
        "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
        "redirects": 1, "titles": title, "format": "json",
    }
    try:
        resp = await _wiki_get(_WIKI_API, params=params)
        pages = resp.json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
    except Exception as exc:  # noqa: BLE001
        return f"error: could not read '{title}' ({exc})"
    if not extract:
        return f"no article text found for '{title}'"
    return extract[:2500]  # cap so one page can't flood the context window


# --- Safe calculator: parse the expression to an AST and evaluate only the
# arithmetic node types we allow. No bare eval() — that would run arbitrary code.
_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate one arithmetic AST node, rejecting anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool
async def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '3776 - 1950' or '(12+8)*3'."""
    try:
        result = _eval_node(ast.parse(expression, mode="eval").body)
    except Exception as exc:  # noqa: BLE001
        return f"error: cannot evaluate '{expression}' ({exc})"
    return f"{expression} = {result}"


ALL_TOOLS: list[BaseTool] = [search_wikipedia, read_wikipedia, calculator]
