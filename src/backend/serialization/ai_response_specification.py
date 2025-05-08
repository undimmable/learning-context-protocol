from typing import override, Protocol

from src.backend.util.util import split_lines


class TokenizationError(Exception):
    def __init__(self, message="Failed to tokenize the AI response"):
        self.message = message
        super().__init__(self.message)


class TokenizerSpecification(Protocol):
    def get_prompt(self) -> str:
        ...


class Token(Protocol):
    def __str__(self) -> str:
        ...

    def get_value(self) -> str:
        ...

    def get_name(self) -> str:
        ...


class AIResponseTokenSpecification:
    def __init__(self, name: str | None, optional: bool = False):
        self.name = name
        self.optional = optional

    def as_partial_prompt(self) -> str:
        return f"{self.name.upper()}: ... " + ("(if any)" if self.optional else "") + "\n"

    def as_readable_line(self, token: Token) -> str:
        return f"{self.name.upper()}: {token.get_value()} + \n"

    def tokenize_line(self, line: str):
        return AIResponseToken(self, line)

    def accepts(self, line: str):
        return line.startswith(self.name.upper() + ":")


class UnspecifiedAIResponseTokenSpecification(AIResponseTokenSpecification):
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
    def as_readable_line(self, token: Token) -> str:
        return token.get_value() + "\n"

    @override
    def tokenize_line(self, line: str) -> Token:
        return UnspecifiedAIResponseToken(self, line)

    def accepts(self, line: str) -> bool:
        return True


class AIResponseToken(Token):
    def __init__(self, spec: AIResponseTokenSpecification, value: str):
        self.spec = spec
        self.value = value

    @override
    def get_name(self) -> str:
        return self.spec.name

    @override
    def get_value(self) -> str:
        return self.value + "\n"


class UnspecifiedAIResponseToken(AIResponseToken):
    def __init__(self, spec: AIResponseTokenSpecification, line: str):
        super().__init__(spec, line)

    @override
    def get_value(self) -> str:
        return self.value + "\n"

    @override
    def __str__(self):
        return self.spec.as_readable_line(self)


class TokenizedAIResponse:
    def __init__(self):
        self.tokens = []

    def add_token(self, token: AIResponseToken):
        self.tokens.append(token)


class AIResponseTokenizer(TokenizerSpecification):
    def __init__(self):
        self.token_specs = [
            AIResponseTokenSpecification("plan", False),
            AIResponseTokenSpecification("state", False),
            AIResponseTokenSpecification("note", True),
            UnspecifiedAIResponseTokenSpecification()
        ]

    def find_spec(self, line: str) -> AIResponseTokenSpecification:
        for token_spec in self.token_specs:
            if token_spec.accepts(line):
                return token_spec
        raise TokenizationError(f"No matching token specification found for line: {line}")

    def tokenize_line(self, line: str) -> AIResponseToken:
        return self.find_spec(line).tokenize_line(line)

    def tokenize(self, raw_ai_response: str) -> TokenizedAIResponse:
        tokenized_ai_response = TokenizedAIResponse()

        for line in split_lines(raw_ai_response):
            tokenized_ai_response.add_token(self.tokenize_line(line))

        return tokenized_ai_response

    @override
    def get_prompt(self) -> str:
        return "".join(token.as_partial_prompt() for token in self.token_specs)
