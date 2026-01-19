import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


class Config(BaseModel):
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None
    OLLAMA_API_KEY: str | None = None
    OLLAMA_HOST: str = "http://localhost:11434"

    @classmethod
    def new(cls, config_path: str | Path | None = None) -> "Config":
        # Input validation
        if not isinstance(config_path, (str, Path)) and config_path is not None:
            raise TypeError(
                "`config_path` expected to be `str`, `Path` or `None`. "
                f"Got `{type(config_path).__name__}` instead."
            )
        elif config_path is None:
            load_dotenv()
        elif isinstance(config_path, str):
            load_dotenv(Path(config_path))
        else:
            load_dotenv(config_path)

        return Config(
            OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY"),
            ANTHROPIC_API_KEY=os.environ.get("ANTHROPIC_API_KEY"),
            GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY"),
            DEEPSEEK_API_KEY=os.environ.get("DEEPSEEK_API_KEY"),
            OLLAMA_API_KEY=os.environ.get("OLLAMA_API_KEY"),
            OLLAMA_HOST=os.environ.get("OLLAMA_HOST") or "http://localhost:11434",
        )
