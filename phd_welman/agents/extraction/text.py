from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, field_validator
from pydantic_ai import Agent, ModelSettings


class TextExtractorInput(BaseModel):
    """Input type for extraction operations. Encapsulates what to extract and
    from where, with optional guidance on extraction criteria."""

    query: str
    "What value or text to extract (e.g., 'email address', 'total amount', 'author name')."

    source: Any
    "Source object, text, or data structure to extract from."

    criteria: str | None = None
    "Optional criteria or constraints for extraction (e.g., 'only numeric values', 'first occurrence')."

    @field_validator("query")
    def validate_query(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be empty")
        return value.strip()

    def to_prompt(self) -> str:
        prompt = f"Extract: {self.query}\nSource: {self.source}"
        if self.criteria:
            prompt += f"\nCriteria: {self.criteria}"
        return prompt


class TextExtractorResult(BaseModel):
    """Output type for extraction operation results. Contains the extracted
    value and reason of how it was found."""

    value: str | None
    "The extracted value or text. Can be None if nothing was found."

    reason: str
    "Explanation of how and where the value was extracted from."

    confidence: Literal["high", "medium", "low"]
    "Confidence level in the extraction accuracy."


def create_direct_text_extraction_agent(
    model: str,
) -> Callable[[TextExtractorInput], Awaitable[TextExtractorResult]]:
    """Create an asynchronous direct extractor agent.

    Args:
        model (str): Provider model identifier passed to pydantic_ai.Agent.

    Returns:
        Callable[[TextExtractorInput], Awaitable[TextExtractorResult]]: Awaitable
        callable that resolves to a literal extraction result.
    """

    agent = Agent(
        name="DirectExtractor",
        model=model,
        output_type=TextExtractorResult,
        system_prompt="""
        You are a DIRECT extractor. Extract values or text EXACTLY as they appear in the source.

        Rules:
        - Extract literal values without interpretation or transformation
        - Preserve exact formatting, capitalization, and spacing
        - If the exact text/value doesn't exist, return None
        - Don't infer or derive values - only extract what's explicitly present
        - For structured data, navigate the exact path to the value

        Examples:

        Extract: email
        Source: {"contact": "john@example.com", "email_type": "work"}
        Result: {"value": null, "explanation": "No key named 'email' exists in source", "confidence": "high"}

        Extract: contact
        Source: {"contact": "john@example.com", "email_type": "work"}
        Result: {"value": "john@example.com", "explanation": "Extracted value from 'contact' key", "confidence": "high"}

        Extract: price
        Source: "The total price is $49.99 after discount"
        Result: {"value": null, "explanation": "No field named 'price' found in text", "confidence": "high"}

        Extract: total amount
        Source: {"total_amount": 1500, "currency": "USD"}
        Result: {"value": null, "explanation": "No exact key 'total amount' exists (found 'total_amount' instead)", "confidence": "high"}

        Now perform the extraction.
        """,
        model_settings=ModelSettings(max_tokens=256, temperature=0.0),
    )

    async def run(model_input: TextExtractorInput) -> TextExtractorResult:
        prompt = f"{model_input.to_prompt()}\nResult: "
        result = await agent.run(prompt)
        return result.output

    return run


def create_semantic_text_extraction_agent(
    model: str,
) -> Callable[[TextExtractorInput], Awaitable[TextExtractorResult]]:
    """Create an asynchronous semantic extraction agent.

    Args:
        model (str): Provider model identifier passed to pydantic_ai.Agent.

    Returns:
        Callable[[TextExtractorInput], Awaitable[TextExtractorResult]]: Awaitable
        callable that resolves to a semantically interpreted extraction result.
    """

    agent = Agent(
        name="SemanticExtractor",
        model=model,
        output_type=TextExtractorResult,
        system_prompt="""
        You are a SEMANTIC extractor. Extract values or text based on their MEANING, not just literal matches.

        Rules:
        - Understand the intent behind the query
        - Match based on semantic meaning, not exact field names
        - Handle synonyms, variations, and related concepts
        - Extract from unstructured text when needed
        - Apply domain knowledge to identify relevant values
        - Transform or parse when it helps achieve the extraction goal

        Examples:

        Extract: email
        Source: {"contact": "john@example.com", "email_type": "work"}
        Result: {"value": "john@example.com", "explanation": "Found email address in 'contact' field", "confidence": "high"}

        Extract: email address
        Source: "Contact John at john.doe@company.org for more info"
        Result: {"value": "john.doe@company.org", "explanation": "Extracted email address from unstructured text", "confidence": "high"}

        Extract: price
        Source: "The total price is $49.99 after discount"
        Result: {"value": "$49.99", "explanation": "Extracted price value from text following 'price is'", "confidence": "high"}

        Extract: total amount
        Source: {"total_amount": 1500, "currency": "USD"}
        Result: {"value": 1500, "explanation": "Matched 'total amount' to 'total_amount' field semantically", "confidence": "high"}

        Extract: author
        Source: {"written_by": "Jane Smith", "publication_date": "2024-01-15"}
        Result: {"value": "Jane Smith", "explanation": "Identified 'written_by' as author field", "confidence": "high"}

        Extract: when was it published
        Source: {"written_by": "Jane Smith", "publication_date": "2024-01-15"}
        Result: {"value": "2024-01-15", "explanation": "Extracted publication date from 'publication_date' field", "confidence": "high"}

        Now perform the extraction.
        """,
        model_settings=ModelSettings(max_tokens=256, temperature=0.0),
    )

    async def run(model_input: TextExtractorInput) -> TextExtractorResult:
        prompt = f"{model_input.to_prompt()}\nResult: "
        result = await agent.run(prompt)
        return result.output

    return run


if __name__ == "__main__":
    import asyncio

    from ...config import Config
    from ...utilities import rprint

    config = Config.new()

    # Test data
    test_data = {
        "user_email": "alice@example.com",
        "total_amount": 1500,
        "currency": "USD",
        "created_by": "Bob Johnson",
    }

    test_text = (
        "Please contact Sarah at sarah.wilson@company.com for the invoice of $299.50"
    )

    rprint("[bold red]# DIRECT EXTRACTOR TESTS[/]\n")

    async def _demo() -> None:
        direct_extractor = create_direct_text_extraction_agent("openai:gpt-4.1-nano")

        # Test 1: Exact key match
        rprint("[bold cyan]Test 1: Exact key match[/]")
        result = await direct_extractor(
            TextExtractorInput(query="total_amount", source=test_data)
        )
        rprint(f"[bold green]Value:[/] [white]{result.value}[/]")
        rprint(f"[bold green]Explanation:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

        # Test 2: Semantic match (should fail for direct)
        rprint("[bold cyan]Test 2: Semantic match (should fail)[/]")
        result = await direct_extractor(
            TextExtractorInput(query="email", source=test_data)
        )
        rprint(f"[bold green]Value:[/] [white]{result.value}[/]")
        rprint(f"[bold green]Explanation:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

        rprint("[bold red]# SEMANTIC EXTRACTOR TESTS[/]\n")

        semantic_extractor = create_semantic_text_extraction_agent(
            "openai:gpt-4.1-nano"
        )

        # Test 3: Semantic match
        rprint("[bold cyan]Test 3: Semantic email extraction[/]")
        result = await semantic_extractor(
            TextExtractorInput(query="email", source=test_data)
        )
        rprint(f"[bold green]Value:[/] [white]{result.value}[/]")
        rprint(f"[bold green]Explanation:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

        # Test 4: Author extraction
        rprint("[bold cyan]Test 4: Author extraction[/]")
        result = await semantic_extractor(
            TextExtractorInput(query="author", source=test_data)
        )
        rprint(f"[bold green]Value:[/] [white]{result.value}[/]")
        rprint(f"[bold green]Explanation:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

        # Test 5: Unstructured text extraction
        rprint("[bold cyan]Test 5: Email from unstructured text[/]")
        result = await semantic_extractor(
            TextExtractorInput(query="email address", source=test_text)
        )
        rprint(f"[bold green]Value:[/] [white]{result.value}[/]")
        rprint(f"[bold green]Explanation:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

        # Test 6: Price extraction from text
        rprint("[bold cyan]Test 6: Price from unstructured text[/]")
        result = await semantic_extractor(
            TextExtractorInput(query="price", source=test_text)
        )
        rprint(f"[bold green]Value:[/] [white]{result.value}[/]")
        rprint(f"[bold green]Explanation:[/] [white]{result.reason}[/]")
        rprint(f"[bold green]Confidence:[/] [white]{result.confidence}[/]\n")

    asyncio.run(_demo())
