import asyncio
import json
import sys

from samsagent.runtime import Runtime
from conftest import EmptyMCP, FakeModel, eventually


async def test_sigkill_during_side_effect_does_not_replay(config):
    driver = config.data_dir / "crash_driver.py"
    driver.write_text('''import asyncio, json, sys
from pathlib import Path
from samsagent.config import Config
from samsagent.runtime import Runtime
class Model:
    async def check(self): pass
    async def chat(self, *args):
        return {"message":{"role":"assistant", "tool_calls":[{"function":{"name":"test__effect", "arguments":{}}}]}}
class Tools:
    tools={"test__effect":{"type":"function","function":{"name":"test__effect","parameters":{"type":"object"}}}}
    async def start(self): pass
    async def close(self): pass
    async def call(self, name, args):
        (root/"effect.txt").write_text("executed once")
        (root/"ready").write_text("ready")
        await asyncio.sleep(60)
async def main():
    runtime=Runtime(Config(ollama_url="unused",model="test",data_dir=root),Model(),Tools())
    await runtime.start()
    session=runtime.create_session("crash recovery",mode="goal")
    (root/"session.json").write_text(json.dumps(session))
    await asyncio.sleep(120)
root=Path(sys.argv[1])
asyncio.run(main())
''')
    process = await asyncio.create_subprocess_exec(sys.executable, str(driver), str(config.data_dir),
                                                   stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        await eventually(lambda: (config.data_dir / "ready").exists())
        process.kill()
        await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    session = json.loads((config.data_dir / "session.json").read_text())

    class Tools(EmptyMCP):
        tools = {"test__effect": {"type": "function", "function": {"name": "test__effect", "parameters": {"type": "object"}}}}

        async def call(self, name, args):
            raise AssertionError("An uncertain effect must not be replayed")

    runtime = Runtime(config, FakeModel(), Tools())
    runtime.store.execute("UPDATE sessions SET status='paused'")
    await runtime.start()
    try:
        agent = runtime.store.one("agents", session["agent_id"])
        uncertain = runtime.store.rows("SELECT * FROM operations WHERE status='uncertain'")
        assert len(uncertain) == 1
        assert runtime.store.history(agent["id"])[-1][1]["role"] == "tool"
        try:
            await runtime.dispatch(agent, "test__effect", {})
        except ValueError as exc:
            assert "uncertain" in str(exc)
        else:
            raise AssertionError("Replay was not blocked")
        assert (config.data_dir / "effect.txt").read_text() == "executed once"
    finally:
        await runtime.close()
