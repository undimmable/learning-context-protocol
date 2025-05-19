from typing import Protocol


class TypedTokenizerSpecification(Protocol):
    def get_prompt(self) -> str:
        ...


class TypedTokenizationError(Exception):
    def __init__(self, message="Failed to tokenize the typed message"):
        self.message = message
        super().__init__(self.message)


class TypedToken(Protocol):
    def __str__(self) -> str:
        ...

    def get_value(self) -> str:
        ...

    def get_name(self) -> str:
        ...
