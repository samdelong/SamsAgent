import asyncio
import json

from samsagent.runtime import Runtime
from samsagent.workflows import create_workflow
from conftest import EmptyMCP, FakeModel, eventually


async def test_workflow_sdk_parallel_children_and_private_prints(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    await runtime.start()
    try:
        session = runtime.create_session("workflows")
        root = runtime.store.one("agents", session["agent_id"])
        source = config.data_dir / "flow.py"
        source.write_text('''import asyncio
async def run(ctx):
    print("ordinary print must not corrupt RPC")
    state = await ctx.checkpoint()
    children = await asyncio.gather(ctx.spawn("A"), ctx.spawn("B"))
    await ctx.checkpoint({"children": children})
    results = await asyncio.gather(*(ctx.wait(child) for child in children))
    summary = await ctx.llm("Summarize the experiment")
    return {"results": results, "summary": summary}
''')
        identifier = create_workflow(runtime, root, source)
        await eventually(lambda: runtime.store.one("workflows", identifier)["status"] in {"completed", "failed"}, 15)
        flow = runtime.store.one("workflows", identifier)
        assert flow["status"] == "completed", flow["result"]
        assert len(json.loads(flow["checkpoint"])["children"]) == 2
        assert all(r["status"] == "completed" for r in json.loads(flow["result"])["results"])
        assert runtime.model.maximum == 1
        assert "ordinary print" in (config.data_dir / "sessions" / session["id"] / "workflows" / identifier / "stderr.log").read_text()
    finally:
        await runtime.close()


async def test_workflow_restarts_from_checkpoint_and_snapshot(config):
    first = Runtime(config, FakeModel(), EmptyMCP())
    await first.start()
    session = first.create_session("restart workflow")
    root = first.store.one("agents", session["agent_id"])
    source = config.data_dir / "resume.py"
    source.write_text('''import asyncio
async def run(ctx):
    state = await ctx.checkpoint()
    if not state:
        await ctx.checkpoint({"phase":"prepared"})
        await asyncio.sleep(60)
    return {"resumed": state["phase"]}
''')
    identifier = create_workflow(first, root, source)
    await eventually(lambda: first.store.one("workflows", identifier)["checkpoint"] != "{}")
    source.write_text("raise RuntimeError('must use snapshot')")
    await first.close()
    second = Runtime(config, FakeModel(), EmptyMCP())
    await second.start()
    try:
        await eventually(lambda: second.store.one("workflows", identifier)["status"] in {"completed", "failed"})
        flow = second.store.one("workflows", identifier)
        assert flow["status"] == "completed", flow["result"]
        assert json.loads(flow["result"]) == {"resumed": "prepared"}
    finally:
        await second.close()


async def test_paused_workflow_waiting_for_inference_resumes_without_failure(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    await runtime.start()
    held = False
    try:
        session = runtime.create_session("workflow pause")
        await eventually(lambda: runtime.store.one("agents", session["agent_id"])["status"] == "idle")
        await runtime.inference.acquire()
        held = True
        source = config.data_dir / "pause.py"
        source.write_text('''async def run(ctx):
    await ctx.checkpoint({"phase": "before_model"})
    return await ctx.llm("hello")
''')
        identifier = create_workflow(runtime, runtime.store.one("agents", session["agent_id"]), source)
        await eventually(lambda: runtime.store.one("workflows", identifier)["checkpoint"] != "{}")
        await runtime.control(session["id"], "pause")
        runtime.inference.release()
        held = False
        await asyncio.sleep(.15)
        assert runtime.store.one("workflows", identifier)["status"] == "running"
        await runtime.control(session["id"], "resume")
        await eventually(lambda: runtime.store.one("workflows", identifier)["status"] in {"completed", "failed"})
        assert runtime.store.one("workflows", identifier)["status"] == "completed"
    finally:
        if held:
            runtime.inference.release()
        await runtime.close()
