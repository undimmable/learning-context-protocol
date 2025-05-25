from typing import override

from src.util.util import split_lines
from src.serialization.tokenizer import TypedTokenizerSpecification, TypedToken, TypedTokenizationError


class AIResponseToken(TypedTokenizerSpecification):
    def __init__(self, name: str | None, optional: bool = False):
        self.name = name
        self.optional = optional

    def as_partial_prompt(self) -> str:
        return f"{self.name.upper()}: ... " + ("(if any)" if self.optional else "") + "\n"

    def as_readable_line(self, token: TypedToken) -> str:
        return f"{self.name.upper()}: {token.get_value()} + \n"

    def tokenize_line(self, line: str):
        return AIResponseTypedToken(self, line)

    def accepts(self, line: str):
        return line.startswith(self.name.upper() + ":")


class UnspecifiedAIResponseToken(AIResponseToken):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            super().__init__(None)
            self.initialized = True

    @override
    def as_partial_prompt(self) -> str:
        return ""

    @override
    def as_readable_line(self, token: TypedToken) -> str:
        return token.get_value() + "\n"

    @override
    def tokenize_line(self, line: str) -> TypedToken:
        return UnspecifiedAIResponseTypedToken(self, line)

    def accepts(self, line: str) -> bool:
        return True


class AIResponseTypedToken(TypedToken):
    def __init__(self, spec: AIResponseToken, value: str):
        self.spec = spec
        self.value = value

    @override
    def get_name(self) -> str:
        return self.spec.name

    @override
    def get_value(self) -> str:
        return self.value + "\n"


class UnspecifiedAIResponseTypedToken(AIResponseTypedToken):
    def __init__(self, spec: AIResponseToken, line: str):
        super().__init__(spec, line)

    @override
    def get_value(self) -> str:
        return self.value + "\n"

    @override
    def __str__(self):
        return self.spec.as_readable_line(self)


class TypedAIResponse:
    def __init__(self):
        self.typed_tokens = []


    def add_typed_token(self, token: AIResponseTypedToken):
        self.typed_tokens.append(token)


class AIResponseTypedTokenizer(TypedTokenizerSpecification):
    def __init__(self):
        self.token_specs = [
            AIResponseToken("plan", False),
            AIResponseToken("state", False),
            AIResponseToken("note", True),
            UnspecifiedAIResponseToken()
        ]

    def find_spec(self, line: str) -> AIResponseToken:
        for token_spec in self.token_specs:
            if token_spec.accepts(line):
                return token_spec
        raise TypedTokenizationError(f"No matching token specification found for line: {line}")

    def tokenize_line(self, line: str) -> AIResponseTypedToken:
        return self.find_spec(line).tokenize_line(line)

    def tokenize(self, raw_ai_response: str) -> TypedAIResponse:
        tokenized_ai_response = TypedAIResponse()

        for line in split_lines(raw_ai_response):
            tokenized_ai_response.add_typed_token(self.tokenize_line(line))

        return tokenized_ai_response

    @override
    def get_prompt(self) -> str:
        return "".join(token.as_partial_prompt() for token in self.token_specs)
