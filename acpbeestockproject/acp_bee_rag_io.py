from typing import Any
from pydantic import BaseModel, Field


class RagInput(BaseModel):
    input: dict[str, Any] = Field(description="User's question and links of websites with answer")


class RagOutput(BaseModel):
    output: dict[str, Any] = Field(description="Answer to user's question")
