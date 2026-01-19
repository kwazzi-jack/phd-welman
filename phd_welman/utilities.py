import json
from typing import Any

from pydantic import BaseModel
from rich import print as rprint


def format_datamodel(model: dict[Any, Any] | BaseModel) -> str:
    """Simple helper function to format data models."""

    if isinstance(model, dict):
        return json.dumps(model, indent=0)
    elif isinstance(model, BaseModel):
        return model.model_dump_json(indent=0)
    raise TypeError(
        f"Expected `model` to be `dict` or `BaseModel`. Got `{type(model).__name__}` instead"
    )


__all__ = ["rprint", "format_datamodel"]
