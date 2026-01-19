import asyncio
import tempfile
from pathlib import Path

from phd_welman.agents.workflow.compile import (
    WorkflowCompileState,
    create_workflow_compile_agent,
)
from phd_welman.utilities import rprint


async def main() -> None:
    workflow_compile_agent = create_workflow_compile_agent()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        (tmp_root / "observation.ms").mkdir()

        state = WorkflowCompileState(
            science_description=f"Process this NGC1482 measurement set please. The measurement set is at {tmp_root}"
        )
        compile_result = await workflow_compile_agent(state)
        rprint("Stage:", compile_result.stage)
        rprint("Reason:", compile_result.reason)
        rprint("Paths:", [str(p) for p in state.ms_paths])
        rprint(state)


if __name__ == "__main__":
    asyncio.run(main())
