import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
import sys

import jsonschema
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .config import MCPServer


class Connection:
    """One owning task enters and exits the SDK's AnyIO transport scopes."""
    def __init__(self, config, timeout):
        self.config, self.timeout = config, timeout
        self.queue = asyncio.Queue()
        self.task = asyncio.create_task(self.run())

    async def request(self, method, *args):
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((method, args, future))
        return await future

    async def run(self):
        while True:
            method, args, future = await self.queue.get()
            try:
                async with AsyncExitStack() as stack:
                    cfg = self.config
                    if cfg.command:
                        transport = stdio_client(StdioServerParameters(
                            command=cfg.command, args=cfg.args, env=cfg.env))
                    else:
                        client = await stack.enter_async_context(httpx.AsyncClient(
                            headers=cfg.headers, timeout=httpx.Timeout(self.timeout)))
                        transport = streamable_http_client(cfg.url, http_client=client)
                    streams = await stack.enter_async_context(transport)
                    session = await stack.enter_async_context(ClientSession(
                        streams[0], streams[1], read_timeout_seconds=timedelta(seconds=self.timeout)))
                    await session.initialize()
                    while True:
                        if not future.cancelled():
                            result = await getattr(session, method)(*args)
                            if not future.done():
                                future.set_result(result)
                        method, args, future = await self.queue.get()
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as exc:
                if not future.done():
                    future.set_exception(RuntimeError(f"MCP connection failed: {exc}"))
                # Reconnect only for the next request; never replay a failed tool call.

    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while not self.queue.empty():
            _, _, future = self.queue.get_nowait()
            if not future.done():
                future.cancel()


class MCPTools:
    def __init__(self, config):
        self.config = config
        self.connections = {}
        self.tools = {}

    async def start(self):
        servers = {"machine": MCPServer(command=sys.executable, args=[
            "-m", "samsagent.machine", "--state-dir", str(self.config.data_dir / "processes")]),
            **self.config.mcp}
        for name, cfg in servers.items():
            if not name.replace("_", "").isalnum():
                raise ValueError("MCP server names must be alphanumeric or underscore")
            self.connections[name] = Connection(cfg, self.config.tool_timeout)
        await self.refresh()

    async def refresh(self):
        for server, connection in self.connections.items():
            cursor = None
            while True:
                result = await connection.request("list_tools", cursor)
                for tool in result.tools:
                    jsonschema.Draft202012Validator.check_schema(tool.inputSchema)
                    name = f"{server}__{tool.name}"
                    self.tools[name] = {"type": "function", "function": {
                        "name": name, "description": tool.description or "",
                        "parameters": tool.inputSchema}}
                cursor = result.nextCursor
                if not cursor:
                    break

    def search(self, query):
        words = query.lower().split()
        candidates = [tool for name, tool in self.tools.items()
                      if name != "machine__process_terminate_owner"]
        return sorted(candidates, key=lambda t: -sum(
            word in str(t).lower() for word in words))[:8]

    async def call(self, name, args):
        schema = self.tools[name]["function"]["parameters"]
        jsonschema.validate(args, schema)
        server, tool = name.split("__", 1)
        result = await self.connections[server].request("call_tool", tool, args)
        return result.model_dump(mode="json", exclude_none=True)

    async def close(self):
        await asyncio.gather(*(conn.close() for conn in self.connections.values()))
