from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from phd_welman.agents.extraction.path import (
    PathExtractionError,
    create_path_extraction_agent,
)


class WorkflowCompileState(BaseModel):
    ms_paths: list[Path] = Field(default_factory=list)
    science_description: str = ""
    use_flagging: bool = False
    use_selfcal: bool = False
    use_transfercal: bool = False
    use_ddecal: bool = False
    use_dirty_image: bool = False
    use_clean_image: bool = False
    output_dir: Path = Field(default_factory=Path.cwd)


@dataclass
class WorkflowCompileResult:
    stage: str
    reason: str | None = None
    payload: WorkflowCompileState | None = None


path_agent = create_path_extraction_agent(
    patterns=["*.ms", "*.MS"],
    types=["directory"],
    permissions=["read"],
    interactive=False,
)


class MSPathExtraction(BaseNode[WorkflowCompileState, None, WorkflowCompileResult]):
    async def run(
        self, ctx: GraphRunContext[WorkflowCompileState, None]
    ) -> End[WorkflowCompileResult]:
        try:
            path_result = await path_agent(ctx.state.science_description)
        except PathExtractionError as error:
            return End(
                WorkflowCompileResult(
                    stage="measurement set path extraction",
                    reason=str(error),
                )
            )

        if not path_result.paths:
            return End(
                WorkflowCompileResult(
                    stage="measurement set path extraction",
                    reason="No measurement set paths could be derived from the description.",
                )
            )

        ctx.state.ms_paths = path_result.paths
        return End(
            WorkflowCompileResult(
                stage="measurement set path extraction", payload=ctx.state
            )
        )


def create_workflow_compile_agent() -> Callable[
    [WorkflowCompileState], Awaitable[WorkflowCompileResult]
]:
    """Build an asynchronous workflow compiler agent.

    Returns:
        Callable[[WorkflowCompileState], Awaitable[WorkflowCompileResult]]: Awaitable
        callable that resolves to the compiled workflow result state.
    """

    graph = Graph(
        name="WorkflowCompileAgent",
        nodes=(MSPathExtraction,),
        state_type=WorkflowCompileState,
    )

    async def run(state: WorkflowCompileState) -> WorkflowCompileResult:
        run_result = await graph.run(MSPathExtraction(), state=state)
        return run_result.output

    return run
