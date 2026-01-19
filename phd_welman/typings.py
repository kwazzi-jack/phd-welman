from dataclasses import dataclass
from typing import Literal

Confidence = Literal["high", "medium", "low"]
FileType = Literal["file", "directory", "symlink", "any"]
FilePermission = Literal["read", "write", "execute"]


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    "The result of the validation check itself."

    error_message: str | None = None
    "If invalid, the error message associated with the invalidation."
