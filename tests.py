import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import phd_welman.agents.boolean as boolean_module
import phd_welman.agents.extraction.text as extraction_text_module
from phd_welman.agents.boolean import BooleanInput, BooleanResult, create_boolean_agent
from phd_welman.agents.extraction.path import (
    PathExtractionError,
    create_path_extraction_agent,
)
from phd_welman.agents.extraction.text import (
    TextExtractorInput,
    TextExtractorResult,
    create_direct_text_extraction_agent,
    create_semantic_text_extraction_agent,
)


class PathExtractionAgentTests(unittest.TestCase):
    def test_empty_patterns_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_path_extraction_agent(patterns=[], interactive=False)

    def test_explicit_path_is_validated(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target = tmp_path / "observation.ms"
            target.touch()
            agent = create_path_extraction_agent(
                patterns=["*.ms"],
                types=["file"],
                permissions=["read"],
                search_locations=[tmp_path],
                interactive=False,
            )
            result = asyncio.run(agent(f"Open the measurement set at {target}"))
            self.assertEqual([target], result.paths)
            self.assertFalse(result.used_auto_search)
            self.assertEqual([], result.diagnostics)

    def test_description_paths_prime_auto_search(self) -> None:
        with TemporaryDirectory() as tmpdir:
            temporary_root = Path(tmpdir)
            described_root = temporary_root / "ms-root"
            described_root.mkdir()
            fallback_file = described_root / "observation.ms"
            fallback_file.touch()
            other_root = temporary_root / "other"
            other_root.mkdir()
            agent = create_path_extraction_agent(
                patterns=["*.ms"],
                types=["file"],
                permissions=["read"],
                search_locations=[other_root],
                interactive=False,
            )
            description = f"Process the measurement set located in {described_root}"
            result = asyncio.run(agent(description))
            self.assertEqual([fallback_file], result.paths)
            self.assertTrue(result.used_auto_search)

    def test_auto_search_fails_without_matching_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            described_root = Path(tmpdir) / "empty"
            described_root.mkdir()
            agent = create_path_extraction_agent(
                patterns=["*.ms"],
                types=["file"],
                permissions=["read"],
                search_locations=[described_root],
                interactive=False,
            )
            description = f"Process the measurement set located in {described_root}"
            with self.assertRaises(PathExtractionError):
                asyncio.run(agent(description))


class FakeResponse:
    def __init__(self, output):
        self.output = output


class StubAgentFactory:
    def __init__(self, expected_output):
        self.expected_output = expected_output

    def __call__(self, *args, **kwargs):
        expected = self.expected_output

        class AgentStub:
            async def run(self, prompt: str) -> FakeResponse:
                return FakeResponse(expected)

        return AgentStub()


class AgentOverride:
    def __init__(self, module, factory):
        self.module = module
        self.factory = factory
        self.original = module.Agent

    def __enter__(self):
        self.module.Agent = self.factory

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.Agent = self.original


class ExternalAgentTests(unittest.TestCase):
    def test_boolean_agent_result_passthrough(self) -> None:
        expected = BooleanResult(value=True, reason="ok", confidence="high")
        factory = StubAgentFactory(expected)
        with AgentOverride(boolean_module, factory):
            agent = create_boolean_agent("openai:test")
            result = asyncio.run(
                agent(BooleanInput(question="Is the sky blue?", target="weather"))
            )
        self.assertIs(result, expected)

    def test_direct_extractor_result_passthrough(self) -> None:
        expected = TextExtractorResult(value=None, reason="ok", confidence="medium")
        factory = StubAgentFactory(expected)
        with AgentOverride(extraction_text_module, factory):
            extractor = create_direct_text_extraction_agent("openai:test")
            result = asyncio.run(
                extractor(TextExtractorInput(query="field", source={"field": "value"}))
            )
        self.assertIs(result, expected)

    def test_semantic_extractor_result_passthrough(self) -> None:
        expected = TextExtractorResult(value="1", reason="ok", confidence="low")
        factory = StubAgentFactory(expected)
        with AgentOverride(extraction_text_module, factory):
            extractor = create_semantic_text_extraction_agent("openai:test")
            result = asyncio.run(
                extractor(TextExtractorInput(query="price", source="$9.99"))
            )
        self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
