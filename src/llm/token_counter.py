from enum import Enum


class TokenCountProvider(str, Enum):
    GEMINI = "gemini"
    BEEKNOEE = "beeknoee"
    MOCK = "mock"
    GENERIC = "generic"


class TokenCounter:
    def count(self, text: str, provider: TokenCountProvider) -> int:
        if provider == TokenCountProvider.MOCK:
            return len(text.split()) if text else 0
        if provider in (TokenCountProvider.GEMINI, TokenCountProvider.BEEKNOEE):
            return len(text) // 4 if text else 0
        return len(text.split()) if text else 0


def count_tokens(text: str, provider: str) -> int:
    return TokenCounter().count(text, TokenCountProvider(provider))
