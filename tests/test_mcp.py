import asyncio
import json
import os
from pathlib import Path
import socket
import sys

import pytest

from samsagent.config import MCPServer
from samsagent.mcp_client import MCPTools, Connection
from conftest import eventually


def unpack(result):
    structured = result.get("structuredContent")
    if structured:
        return structured.get("result", structured)
    return json.loads(result["content"][0]["text"])


async def test_real_stdio_files_large_results_and_process_lifecycle(config):
    tools = MCPTools(config)
    await tools.start()
    try:
        file = config.data_dir / "large.txt"
        text = "research evidence\n" * 10000
        await tools.call("machine__file_write", {"path": str(file), "content": text})
        result = unpack(await tools.call("machine__file_read", {"path": str(file), "limit": 60000}))
        assert result["text"] == text[:60000] and result["next_offset"] == 60000
        found = unpack(await tools.call("machine__file_search", {"path": str(file), "query": "evidence", "max_matches": 2}))
        assert len(found) == 2
        job = unpack(await tools.call("machine__process_start", {"command": "printf 'experiment result'", "cwd": str(config.data_dir)}))
        for _ in range(100):
            status = unpack(await tools.call("machine__process_status", {"job_id": job["id"]}))
            if status["status"] == "completed":
                break
            await asyncio.sleep(.02)
        assert status["returncode"] == 0
        assert unpack(await tools.call("machine__process_output", {"job_id": job["id"]}))["text"] == "experiment result"
        job = unpack(await tools.call("machine__process_start", {"command": "sleep 30", "cwd": str(config.data_dir), "owner": "test"}))
        stopped = unpack(await tools.call("machine__process_terminate", {"job_id": job["id"]}))
        assert stopped["status"] == "cancelled"
        assert any(t["function"]["name"] == "machine__file_read" for t in tools.search("file read"))
    finally:
        await tools.close()


async def test_streamable_http_and_disconnect_does_not_replay(config, tmp_path):
    server_file = tmp_path / "server.py"
    # A real SDK HTTP server with a tool that drops its connection after recording an effect.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_file.write_text(f'''from mcp.server.fastmcp import FastMCP
from pathlib import Path
import os
server=FastMCP("test", host="127.0.0.1", port={port})
@server.tool()
def echo(text: str) -> str:
    return text
@server.tool()
def disconnect() -> str:
    Path({str(tmp_path / 'effect')!r}).write_text("once")
    os._exit(0)
server.run(transport="streamable-http")
''')
    process = await asyncio.create_subprocess_exec(sys.executable, str(server_file), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    connection = Connection(MCPServer(url=f"http://127.0.0.1:{port}/mcp"), 2)
    try:
        for _ in range(50):
            try:
                result = await connection.request("list_tools")
                break
            except RuntimeError:
                await asyncio.sleep(.1)
        assert {t.name for t in result.tools} == {"echo", "disconnect"}
        result = await connection.request("call_tool", "echo", {"text": "hello"})
        assert "hello" in str(result)
        with pytest.raises(RuntimeError):
            await connection.request("call_tool", "disconnect", {})
        assert (tmp_path / "effect").read_text() == "once"
        await process.wait()
        # The next request can establish a fresh connection after the server returns.
        process = await asyncio.create_subprocess_exec(sys.executable, str(server_file), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        for _ in range(50):
            try:
                result = await connection.request("list_tools")
                break
            except RuntimeError:
                await asyncio.sleep(.1)
        assert len(result.tools) == 2
    finally:
        await connection.close()
        if process.returncode is None:
            process.terminate()
        await process.wait()
