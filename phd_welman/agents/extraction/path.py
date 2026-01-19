from __future__ import annotations

import fnmatch
import glob
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit.shortcuts import prompt
from pydantic import BaseModel, Field, field_validator
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from phd_welman.typings import FilePermission, FileType, ValidationResult

_DEFAULT_FILE_TYPES: tuple[FileType, ...] = ("any",)
_DEFAULT_PERMISSIONS: tuple[FilePermission, ...] = ("read",)

_PERMISSION_FLAGS: dict[FilePermission, int] = {
    "read": os.R_OK,
    "write": os.W_OK,
    "execute": os.X_OK,
}
_MAX_SEARCH_RESULTS = 25
_ABSOLUTE_PATH_REGEX = re.compile(r"(?:/|[A-Za-z]:\\)[^\s'\"<>]+")
_RELATIVE_PATH_REGEX = re.compile(r"(?:\./|\.\./)[\w./-]+")


def _trim_delimiters(value: str) -> str:
    value = value.strip()
    return value.strip("\"'<>[](){}.,;:")


def _has_wildcard(value: str) -> bool:
    return any(ch in value for ch in "*?[]")


def _expand_path_expression(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value.strip()))


def _parse_description_for_paths(description: str) -> list[str]:
    seen: dict[str, None] = {}
    for regex in (_ABSOLUTE_PATH_REGEX, _RELATIVE_PATH_REGEX):
        for match in regex.finditer(description):
            candidate = _trim_delimiters(match.group(0))
            if candidate and candidate not in seen:
                seen[candidate] = None
    return list(seen.keys())


def _expand_candidate_paths(token: str) -> list[Path]:
    cleaned = _expand_path_expression(_trim_delimiters(token))
    if not cleaned:
        return []
    if _has_wildcard(cleaned):
        matches = glob.glob(cleaned, recursive=True)
        return [Path(path) for path in matches if path]
    return [Path(cleaned)]


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        normalized = str(path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(path)
    return ordered


def _extract_keywords(description: str) -> list[str]:
    tokens = re.findall(r"\b[0-9A-Za-z]{4,}\b", description.lower())
    seen: dict[str, None] = {}
    for token in tokens:
        if token not in seen:
            seen[token] = None
        if len(seen) >= 5:
            break
    return list(seen.keys())


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    if not pattern or not pattern.strip():
        raise ValueError("Patterns cannot be empty")
    trimmed = pattern.strip()
    if _has_wildcard(trimmed):
        return re.compile(fnmatch.translate(trimmed))
    return re.compile(trimmed)


def _matches_patterns(path: Path, matchers: Sequence[re.Pattern[str]]) -> bool:
    text = str(path)
    for matcher in matchers:
        if matcher.fullmatch(text):
            return True
    return False


def _check_file_type(path: Path, allowed: set[FileType]) -> tuple[bool, str | None]:
    if "any" in allowed:
        return True, None
    if "file" in allowed and path.is_file():
        return True, None
    if "directory" in allowed and path.is_dir():
        return True, None
    if "symlink" in allowed and path.is_symlink():
        return True, None
    hint = ", ".join(sorted(allowed))
    return False, f"Path {path} is not one of the configured types ({hint})"


def _check_permissions(
    path: Path, permissions: set[FilePermission]
) -> tuple[bool, str | None]:
    for permission in permissions:
        flag = _PERMISSION_FLAGS[permission]
        if not os.access(path, flag):
            return False, f"Path {path} lacks {permission} permission"
    return True, None


def _validate_candidate(
    path: Path, deps: PathExtractionDeps
) -> tuple[bool, str | None]:
    if not _matches_patterns(path, deps.pattern_matchers):
        return False, f"Path {path} does not match any of the configured patterns"
    if not path.exists():
        return False, f"Path {path} does not exist"
    valid_type, type_reason = _check_file_type(path, deps.allowed_types)
    if not valid_type:
        return False, type_reason
    valid_permissions, perm_reason = _check_permissions(path, deps.allowed_permissions)
    if not valid_permissions:
        return False, perm_reason
    for validator in deps.validators:
        result = validator(path)
        if not result.valid:
            return False, result.error_message or f"Custom validator rejected {path}"
    return True, None


def _keyword_matches(path: Path, keywords: Sequence[str]) -> bool:
    if not keywords:
        return True
    name = path.name.lower()
    return any(keyword in name for keyword in keywords)


def _search_for_candidates(
    deps: PathExtractionDeps,
    keywords: Sequence[str],
    extra_locations: Sequence[Path] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    locations: list[Path] = list(deps.search_locations)
    if extra_locations:
        locations.extend(extra_locations)
    for root in _unique_paths(locations):
        if not root.exists():
            continue
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            if depth > deps.search_depth:
                continue
            try:
                for child in current.iterdir():
                    if not child.exists():
                        continue
                    if child.is_dir() and depth < deps.search_depth:
                        stack.append((child, depth + 1))
                    if not _matches_patterns(child, deps.pattern_matchers):
                        continue
                    if not _keyword_matches(child, keywords):
                        continue
                    normalized = str(child)
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    candidates.append(child)
                    if len(candidates) >= _MAX_SEARCH_RESULTS:
                        return sorted(candidates)
            except PermissionError:
                continue
    return sorted(candidates)


def _prompt_user_selection(candidates: Sequence[Path]) -> list[Path]:
    if not candidates:
        return []
    choices = "\n".join(f"{idx + 1}. {path}" for idx, path in enumerate(candidates))
    message = (
        f"Select indexes for the desired paths (comma separated).\n{choices}\n"
        "Press enter to accept the first item."
    )
    try:
        response = prompt(message, default="1")
    except (EOFError, KeyboardInterrupt):
        return []
    chosen: list[Path] = []
    for token in response.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index = int(token) - 1
        except ValueError:
            continue
        if 0 <= index < len(candidates):
            candidate = candidates[index]
            if candidate not in chosen:
                chosen.append(candidate)
    return chosen


@dataclass
class PathExtractionState:
    description: str
    candidate_paths: list[Path] = field(default_factory=list)
    valid_paths: list[Path] = field(default_factory=list)
    invalid_reasons: list[str] = field(default_factory=list)
    search_candidates: list[Path] = field(default_factory=list)
    used_auto_search: bool = False
    keywords: list[str] = field(default_factory=list)
    description_paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class PathExtractionDeps:
    pattern_matchers: tuple[re.Pattern[str], ...]
    allowed_types: set[FileType]
    allowed_permissions: set[FilePermission]
    validators: tuple[Callable[[Path], ValidationResult], ...]
    search_locations: tuple[Path, ...]
    search_depth: int
    auto_search: bool
    interactive: bool


class PathExtractionError(RuntimeError):
    """Indicates the agent could not produce a valid path."""


class PathExtractionInput(BaseModel):
    """Describes the natural language input used by the path extraction agent.

    Attributes:
        description (str): Text prompt describing the files to retrieve.
        context (dict[str, Any] | None): Optional metadata that can influence parsing.
    """

    description: str
    context: dict[str, Any] | None = None

    @field_validator("description")
    def _validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Description cannot be empty")
        return value.strip()


class PathExtractionResult(BaseModel):
    """Captures the validated paths and any diagnostic feedback.

    Attributes:
        paths (list[Path]): Paths that satisfied the configured validators.
        diagnostics (list[str]): Helper text explaining why other candidates failed.
        used_auto_search (bool): Whether the result originated from a search.
    """

    paths: list[Path]
    diagnostics: list[str] = Field(default_factory=list)
    used_auto_search: bool


class ExtractCandidates(
    BaseNode[PathExtractionState, PathExtractionDeps, PathExtractionResult]
):
    """Initial node that parses the description for explicit path tokens."""

    async def run(
        self, ctx: GraphRunContext[PathExtractionState, PathExtractionDeps]
    ) -> ValidateCandidates | AutoSearch | End[PathExtractionResult]:
        text = ctx.state.description
        ctx.state.keywords = _extract_keywords(text)
        expanded: list[Path] = []
        for token in _parse_description_for_paths(text):
            expanded.extend(_expand_candidate_paths(token))
        normalized = _unique_paths(expanded)
        ctx.state.candidate_paths = normalized
        ctx.state.description_paths = list(normalized)
        if ctx.state.candidate_paths:
            return ValidateCandidates()
        if ctx.deps.auto_search:
            return AutoSearch()
        raise PathExtractionError("No explicit paths found and auto search is disabled")


class ValidateCandidates(
    BaseNode[PathExtractionState, PathExtractionDeps, PathExtractionResult]
):
    """Checks the candidates against patterns, type requirements, permissions, and validators."""

    async def run(
        self, ctx: GraphRunContext[PathExtractionState, PathExtractionDeps]
    ) -> Finalize | AutoSearch:
        paths = _unique_paths(ctx.state.candidate_paths)
        ctx.state.valid_paths.clear()
        ctx.state.invalid_reasons.clear()
        for path in paths:
            valid, reason = _validate_candidate(path, ctx.deps)
            if valid:
                ctx.state.valid_paths.append(path)
            elif reason:
                ctx.state.invalid_reasons.append(reason)
        if ctx.state.valid_paths:
            return Finalize()
        if ctx.deps.auto_search:
            return AutoSearch()
        raise PathExtractionError("Unable to validate any candidate paths")


class AutoSearch(
    BaseNode[PathExtractionState, PathExtractionDeps, PathExtractionResult]
):
    """Performs a directory search when explicit candidates fail."""

    async def run(
        self, ctx: "GraphRunContext[PathExtractionState, PathExtractionDeps]"
    ) -> "InteractiveSelection | ValidateCandidates":
        ctx.state.used_auto_search = True
        ctx.state.search_candidates = _search_for_candidates(
            ctx.deps, ctx.state.keywords
        )
        if not ctx.state.search_candidates and ctx.state.description_paths:
            ctx.state.search_candidates = _search_for_candidates(
                ctx.deps,
                (),
                extra_locations=ctx.state.description_paths,
            )
        if not ctx.state.search_candidates:
            raise PathExtractionError("Auto search did not find any matching files")
        if ctx.deps.interactive:
            return InteractiveSelection()
        ctx.state.candidate_paths = ctx.state.search_candidates
        return ValidateCandidates()


class InteractiveSelection(
    BaseNode[PathExtractionState, PathExtractionDeps, PathExtractionResult]
):
    """Asks the user to pick from auto search candidates when interactivity is enabled."""

    async def run(
        self, ctx: "GraphRunContext[PathExtractionState, PathExtractionDeps]"
    ) -> "ValidateCandidates":
        selection = _prompt_user_selection(ctx.state.search_candidates[:5])
        if not selection:
            raise PathExtractionError("Interactive selection did not yield a choice")
        ctx.state.candidate_paths = selection
        return ValidateCandidates()


class Finalize(BaseNode[PathExtractionState, PathExtractionDeps, PathExtractionResult]):
    """Final node that packages validated paths into the result."""

    async def run(
        self, ctx: "GraphRunContext[PathExtractionState, PathExtractionDeps]"
    ) -> End[PathExtractionResult]:
        result = PathExtractionResult(
            paths=list(ctx.state.valid_paths),
            diagnostics=list(ctx.state.invalid_reasons),
            used_auto_search=ctx.state.used_auto_search,
        )
        return End(result)


def _build_path_extraction_graph() -> Graph[
    PathExtractionState, PathExtractionDeps, PathExtractionResult
]:
    return Graph(
        nodes=(
            ExtractCandidates,
            ValidateCandidates,
            AutoSearch,
            InteractiveSelection,
            Finalize,
        )
    )


def create_path_extraction_agent(
    patterns: Sequence[str],
    types: Sequence[FileType] | None = None,
    permissions: Sequence[FilePermission] | None = None,
    validators: Sequence[Callable[[Path], ValidationResult]] | None = None,
    search_locations: Sequence[Path] | None = None,
    search_depth: int = 3,
    auto_search: bool = True,
    interactive: bool = True,
) -> Callable[[str], Awaitable[PathExtractionResult]]:
    """Builds a configured path extraction agent that returns validated paths.

    Args:
        patterns (Sequence[str]): Glob or regex patterns that candidate paths must match.
        types (Sequence[FileType] | None): Allowed file object types, defaults to any.
        permissions (Sequence[FilePermission] | None): Required permissions, defaults to read.
        validators (Sequence[Callable[[Path], ValidationResult]] | None): Additional validators to run after filesystem checks.
        search_locations (Sequence[Path] | None): Directories to explore when auto search runs.
        search_depth (int): Max recursion depth used by the auto search behavior.
        auto_search (bool): Enables automatic search when parsing yields no validated paths.
        interactive (bool): Prompts the user to select a candidate when auto search runs.

    Returns:
        Callable[[str], Awaitable[PathExtractionResult]]: Awaitable callable that
        executes the configured graph and returns the validated paths.

    Raises:
        ValueError: If the provided configuration is invalid.
    """
    if not patterns:
        raise ValueError("At least one pattern must be supplied")
    trimmed_patterns = tuple(pattern.strip() for pattern in patterns if pattern.strip())
    if not trimmed_patterns:
        raise ValueError("Patterns cannot be empty")
    matchers = tuple(_compile_pattern(pattern) for pattern in trimmed_patterns)
    allowed_types: set[FileType] = set(types or _DEFAULT_FILE_TYPES)
    invalid_type = allowed_types.difference({"file", "directory", "symlink", "any"})
    if invalid_type:
        raise ValueError(f"Unsupported file types: {invalid_type}")
    allowed_permissions: set[FilePermission] = set(permissions or _DEFAULT_PERMISSIONS)
    if invalid_permissions := allowed_permissions - set(_PERMISSION_FLAGS.keys()):
        raise ValueError(f"Unsupported permissions: {invalid_permissions}")
    validator_tuple = tuple(validators or ())
    if search_depth < 0:
        raise ValueError("search_depth must be non negative")
    default_locations = (Path.cwd(), Path.home() / "data")
    normalized_locations = tuple(
        Path(location).expanduser()
        for location in (search_locations or default_locations)
    )
    deps = PathExtractionDeps(
        pattern_matchers=matchers,
        allowed_types=allowed_types,
        allowed_permissions=allowed_permissions,
        validators=validator_tuple,
        search_locations=normalized_locations,
        search_depth=search_depth,
        auto_search=auto_search,
        interactive=interactive,
    )
    graph = _build_path_extraction_graph()

    async def run(description: str) -> PathExtractionResult:
        payload = PathExtractionInput(description=description)
        state = PathExtractionState(description=payload.description)
        run_result = await graph.run(ExtractCandidates(), state=state, deps=deps)
        return run_result.output

    return run


if __name__ == "__main__":
    import asyncio
    import tempfile

    async def _demo() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            for i in range(10):
                (tmp_root / f"dirty-{i}.fits").touch()
                (tmp_root / f"clean-{i}.fits").touch()

            agent = create_path_extraction_agent(
                patterns=["*.fits"],
                types=["file"],
                permissions=["read"],
                search_locations=[tmp_root],
                interactive=False,
            )
            description = f"Please open the fits files at {tmp_root}"
            result = await agent(description)
            print("Result paths:", result.paths)
            print("Diagnostics:", result.diagnostics)
            print("Used auto search:", result.used_auto_search)

    asyncio.run(_demo())
