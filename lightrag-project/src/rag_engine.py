import os
from functools import partial
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.base import EmbeddingFunc
from src.tokenizer import make_tokenizer

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
WORKING_DIR = "rag_storage"

def build_rag() -> LightRAG:
    llm_func = partial(
        gemini_model_complete,
        api_key=GEMINI_API_KEY,
    )

    embed_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=2048,
        func=partial(gemini_embed.func, api_key=GEMINI_API_KEY),
    )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_func,
        llm_model_name=LLM_MODEL,
        embedding_func=embed_func,
        tokenizer=make_tokenizer(),
    )
    return rag