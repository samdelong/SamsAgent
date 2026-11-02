"""An agent may copy/adapt this file and run it with start_workflow.

The two workers gather evidence and execute an experiment, then a synthesis worker
produces the report. Goal completion still requires the runtime's separate reviewer.
"""
import asyncio


async def run(ctx):
    state = await ctx.checkpoint()
    if "sources" not in state:
        state["sources"] = await ctx.spawn(
            "Investigate Python math.fsum versus sum using the official Python docs. "
            "Fetch https://docs.python.org/3/library/math.html using shell tools if needed. "
            "Save source excerpts and URLs in your working directory and return their absolute paths.")
        await ctx.checkpoint(state)
    if "experiment" not in state:
        state["experiment"] = await ctx.spawn(
            "Write and run a reproducible Python experiment comparing sum and math.fsum on "
            "[1e16, 1.0, -1e16] and randomized orderings. Save code, command and output. "
            "Return absolute artifact paths and conclusions, including limitations.")
        await ctx.checkpoint(state)

    async def collect(identifier):
        while True:
            result = await ctx.wait(identifier)
            if result["status"] in {"completed", "blocked", "failed", "cancelled"}:
                return result

    if "results" not in state:
        state["results"] = await asyncio.gather(collect(state["sources"]), collect(state["experiment"]))
        await ctx.checkpoint(state)
    if any(r["status"] != "completed" for r in state["results"]):
        return {"blocked": state["results"]}
    if "synthesis" not in state:
        state["synthesis"] = await ctx.spawn(
            "Read these worker results and inspect their evidence files. Write a Markdown report "
            "answering when math.fsum helps, citing the official docs and experiment results. "
            "Return the report's absolute path.\n" + str(state["results"]))
        await ctx.checkpoint(state)
    return await collect(state["synthesis"])
