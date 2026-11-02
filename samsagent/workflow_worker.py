"""Workflow child and small async SDK. stdout is reserved for private RPC."""
import asyncio
import importlib.util
import json
import os
import sys


class WorkflowContext:
    def __init__(self, wire):
        self.wire = wire
        self.pending = {}
        self.sequence = 0
        self.owner = asyncio.current_task()

    async def receive(self):
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                for future in self.pending.values():
                    if not future.done():
                        future.set_exception(RuntimeError("Harness disconnected"))
                self.owner.cancel()
                return
            message = json.loads(line)
            future = self.pending.pop(message["id"], None)
            if future and not future.done():
                if "error" in message:
                    future.set_exception(RuntimeError(message["error"]))
                else:
                    future.set_result(message.get("result"))

    async def call(self, method, **args):
        self.sequence += 1
        identifier = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        self.wire.write(json.dumps({"id": identifier, "method": method, "args": args}) + "\n")
        self.wire.flush()
        return await future

    async def spawn(self, task, references=None):
        result = await self.call("spawn", task=task, references=references or [])
        return result["agent_id"]

    async def wait(self, agent_id):
        return await self.call("wait", agent_id=agent_id)

    async def message(self, agent_id, text):
        return await self.call("agent_message", agent_id=agent_id, text=text)

    async def llm(self, prompt):
        return await self.call("llm", prompt=prompt)

    async def tool(self, name, arguments):
        return await self.call("tool", name=name, arguments=arguments)

    async def checkpoint(self, state=None):
        return await self.call("checkpoint", state=state)


async def run(path, wire):
    context = WorkflowContext(wire)
    reader = asyncio.create_task(context.receive())
    try:
        spec = importlib.util.spec_from_file_location("agent_workflow", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = await module.run(context)
        wire.write(json.dumps({"done": True, "result": result}) + "\n")
        wire.flush()
    except Exception as exc:
        wire.write(json.dumps({"done": True, "error": repr(exc)}) + "\n")
        wire.flush()
    finally:
        reader.cancel()
        # The parent closes stdin on done, releasing the blocking reader thread.


if __name__ == "__main__":
    if sys.platform == "linux":
        import ctypes
        import signal
        # Kill even non-cooperative generated code when its harness dies on Linux.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot set workflow parent-death signal")
    if os.getppid() != int(sys.argv[2]):
        sys.exit("Harness exited before workflow startup")
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    asyncio.run(run(sys.argv[1], wire))

