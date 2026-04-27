from lightrag.utils import Tokenizer


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


def make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char-utf8", tokenizer=_CharTokenizer())