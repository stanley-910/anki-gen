from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


class BasicCard(BaseModel):
    type: Literal["basic"] = "basic"
    front: str
    back: str
    reversed: bool = False
    tags: list[str] = []


class DefinitionCard(BaseModel):
    type: Literal["definition"] = "definition"
    term: str
    definition: str
    tags: list[str] = []


Card = Union[BasicCard, DefinitionCard]
