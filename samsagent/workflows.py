import ast
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

from .store import atomic_write, uid


def create_workflow(runtime, agent, path):
    source = path.read_text()
    tree = ast.parse(source)
    if not any(isinstance(n, ast.AsyncFunctionDef) and n.name == "run" for n in tree.body):
        raise ValueError("Workflow must define async def run(ctx)")
    identifier = uid()
    snapshot = runtime.store.session_dir(agent["session_id"]) / "workflows" / identifier / "source.py"
    atomic_write(snapshot, source)
    runtime.store.execute("INSERT INTO workflows(id,agent_id,session_id,status,source) VALUES(?,?,?,?,?)",
                          (identifier, agent["id"], agent["session_id"], "ready", str(snapshot)))
    runtime.store.event(agent["session_id"], "workflow_created", {"id": identifier, "agent": agent["id"], "source": str(snapshot)})
    return identifier


async def run_workflow(runtime, identifier):
    from .runtime import Suspended
    store = runtime.store
    flow = store.one("workflows", identifier)
    agent = store.one("agents", flow["agent_id"])
    source = Path(flow["source"])
    process = None
    requests = set()
    writer_lock = asyncio.Lock()

    async def active():
        while store.one("sessions", flow["session_id"])["status"] == "paused":
            await asyncio.sleep(.1)
        runtime.check_active(agent)

    async def execute(message):
        method, args = message["method"], message.get("args", {})
        if method == "checkpoint":
            state = args.get("state")
            if state is not None:
                if not isinstance(state, dict) or len(json.dumps(state)) > 100000:
                    raise ValueError("Checkpoint must be a JSON object under 100KB")
                store.execute("UPDATE workflows SET checkpoint=? WHERE id=?", (json.dumps(state), identifier))
            return json.loads(store.one("workflows", identifier)["checkpoint"])
        if method == "llm":
            prompt = args["prompt"]
            if not isinstance(prompt, str):
                raise ValueError("prompt must be text")
            messages = [{"role": "user", "content": prompt}]
            if runtime.context.estimate(messages) + runtime.config.output_tokens > runtime.config.context_tokens:
                raise ValueError("Workflow prompt exceeds context budget; split or summarize it")
            response = await runtime.infer(agent, messages, [])
            return response["message"].get("content", "")
        if method == "tool":
            if args["name"] not in runtime.mcp.tools:
                raise ValueError("ctx.tool expects an MCP tool name")
            return await runtime.dispatch(agent, args["name"], args["arguments"], identifier)
        if method in {"spawn", "wait", "agent_message"}:
            return await runtime.dispatch(agent, method, args, identifier)
        raise ValueError("Unknown workflow method")

    async def handle(message):
        reply = {"id": message.get("id")}
        while True:
            try:
                await active()
                reply["result"] = await execute(message)
            except Suspended:
                if store.one("sessions", flow["session_id"])["status"] == "paused":
                    continue
                reply["error"] = "Session stopped"
            except Exception as exc:
                reply["error"] = str(exc)
            break
        async with writer_lock:
            if process.stdin and not process.stdin.is_closing():
                process.stdin.write((json.dumps(reply) + "\n").encode())
                await process.stdin.drain()

    try:
        await active()
        store.execute("UPDATE workflows SET status='running' WHERE id=?", (identifier,))
        with open(source.parent / "stderr.log", "ab") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "samsagent.workflow_worker", str(source), str(os.getpid()),
                cwd=store.agent_dir(agent["id"]), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=log, start_new_session=True, limit=1048576)
            runtime.workflow_processes[identifier] = process
            while line := await process.stdout.readline():
                message = json.loads(line)
                if message.get("done"):
                    if requests:
                        await asyncio.gather(*requests)
                    status = "failed" if "error" in message else "completed"
                    result = json.dumps(message.get("error", message.get("result")))
                    store.execute("UPDATE workflows SET status=?,result=? WHERE id=?", (status, result, identifier))
                    store.event(flow["session_id"], "workflow_status", {"id": identifier, "status": status, "result": result})
                    if store.one("agents", agent["id"])["status"] != "cancelled":
                        runtime.send(agent["id"], f"Workflow {identifier} {status}: {result[:4000]}")
                    break
                task = asyncio.create_task(handle(message))
                requests.add(task)
                task.add_done_callback(requests.discard)
            else:
                raise RuntimeError("Workflow exited without a result; inspect stderr.log and uncertain operations")
    except asyncio.CancelledError:
        if store.one("workflows", identifier)["status"] != "cancelled":
            store.execute("UPDATE workflows SET status='ready' WHERE id=?", (identifier,))
        raise
    except Suspended:
        store.execute("UPDATE workflows SET status='ready' WHERE id=? AND status<>'cancelled'", (identifier,))
    except Exception as exc:
        store.execute("UPDATE workflows SET status='failed',result=? WHERE id=?", (str(exc), identifier))
        store.event(flow["session_id"], "workflow_error", {"id": identifier, "error": str(exc)})
        if store.one("agents", agent["id"])["status"] != "cancelled":
            runtime.send(agent["id"], f"Workflow {identifier} failed: {exc}")
    finally:
        async def cleanup():
            for task in requests:
                task.cancel()
            await asyncio.gather(*requests, return_exceptions=True)
            if process:
                if process.stdin:
                    process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except TimeoutError:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
            runtime.workflow_processes.pop(identifier, None)

        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise
