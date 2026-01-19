from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, field_validator
from pydantic_ai import Agent, ModelSettings


class BooleanInput(BaseModel):
    """Input type for a boolean operation. Encapsulates a question and optional
    target material to be examined when evaluating the boolean result."""

    question: str
    "The true/false or yes/no question to ask."

    target: Any | None = None
    "Optional target object, values or text to examine alongside the question."

    @field_validator("question")
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()

    def to_prompt(self) -> str:
        prompt = f"Q: {self.question}"
        if self.target is not None:
            prompt += f"\nTarget: '{self.target}'"
        return prompt


class BooleanResult(BaseModel):
    """Output type for a boolean operation result. Indicates the boolean
    result value and the reason for selecting this result."""

    value: bool
    "Boolean result value. Either True or False."

    reason: str
    "Reason for selecting this boolean result value."

    confidence: Literal["high", "medium", "low"]
    "Confidence level in the extraction accuracy."


def create_boolean_agent(model: str) -> Callable[
    [BooleanInput], Awaitable[BooleanResult]
]:
    """Create an asynchronous boolean agent closure.

    Args:
        model (str): Provider-specific model identifier used by the agent runtime.

    Returns:
        Callable[[BooleanInput], Awaitable[BooleanResult]]: Awaitable function that
        executes the configured agent and resolves to a BooleanResult instance.
    """
    agent = Agent(
        name="BooleanAgent",
        model=model,
        output_type=BooleanResult,
        system_prompt="""
        Examine the provided boolean question and optional target context. "
        Provide a True/False answer with clear reasoning. "

        Here are some examples of the expected format:"
        Q: Is water wet? Target: None
        A: {"value": true, "reason": "Water exhibits wetness through hydrogen bonding", "confidence": "high"}

        Q: Does this describe a mammal: "Has scales, lays eggs, lives underwater"?
        A: {"value": false, "reason": "Description matches fish/reptiles, not mammals", "confidence": "high"}

        Q: Is the distance between these points equal to 5? Target: [Point(-1, 3), Point(3, 0)]
        A: {"value": true, "reason": "Using the distance formula: sqrt((3-(-1))^2 + (0-3)^2) = sqrt(16 + 9) = sqrt(25) = 5", "confidence": "high"}

        Now answer the user's question.
        """,
        model_settings=ModelSettings(max_tokens=128, temperature=0.0),
    )

    async def run(model_input: BooleanInput) -> BooleanResult:
        prompt = f"{model_input.to_prompt()}\nA: "
        result = await agent.run(prompt)
        return result.output

    return run


if __name__ == "__main__":
    import asyncio

    from ..config import Config
    from ..utilities import rprint

    config = Config.new()
    async def _demo() -> None:
        rprint("[bold red]# SIMPLE TEST[/]")
        bool_agent = create_boolean_agent("openai:gpt-4.1-nano")
        question = "Is the sky blue because of refraction effects?"

        rprint(f"[bold yellow]Question:[/] [white]{question}[/]")
        result = await bool_agent(BooleanInput(question=question))
        rprint(f"[bold green]Result:[/] [orange]{result.value}[/]")
        rprint(f"[bold green]Reason:[/] [white]{result.reason}[/]\n")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]")

        rprint("[bold red]# TARGET TEST[/]")
        bool_agent = create_boolean_agent("openai:gpt-4.1-nano")
        question = (
            "Does this description of a science case describe the independent variables?"
        )
        target = (
            "The experiment on the local meerkat population measured how large the population "
            "can be for the Kalagadi ecosystem."
        )
        rprint(f"[bold yellow]Question:[/] [white]{question}[/]")
        rprint(f"[bold yellow]Target:[/] [white]{target}[/]")
        result = await bool_agent(BooleanInput(question=question, target=target))
        rprint(f"[bold green]Result:[/] [orange]{result.value}[/]")
        rprint(f"[bold green]Reason:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

        rprint("[bold red]# LOW CONFIDENCE TEST[/]")
        bool_agent = create_boolean_agent("openai:gpt-4.1-nano")
        question = "I am facing north currently?"
        rprint(f"[bold yellow]Question:[/] [white]{question}[/]")
        rprint(f"[bold yellow]Target:[/] [white]{None}[/]")
        result = await bool_agent(BooleanInput(question=question))
        rprint(f"[bold green]Result:[/] [orange]{result.value}[/]")
        rprint(f"[bold green]Reason:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

    asyncio.run(_demo())
