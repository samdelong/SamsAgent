import asyncio

import pytest

from samsagent.config import Config


def response(content="", calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"function": {"name": name, "arguments": args}} for name, args in calls]
    return {"message": message, "prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"}


class FakeModel:
    def __init__(self, callback=None):
        self.callback = callback or (lambda messages, tools: response("Finished"))
        self.calls = []
        self.active = 0
        self.maximum = 0

    async def check(self):
        pass

    async def chat(self, messages, tools=None, output_tokens=None):
        self.calls.append((messages, tools))
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(.001)
            result = self.callback(messages, tools)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            self.active -= 1


class EmptyMCP:
    tools = {}

    async def start(self):
        pass

    async def close(self):
        pass

    def search(self, query):
        return []

    async def call(self, name, args):
        return {}


@pytest.fixture
def config(tmp_path):
    return Config(ollama_url="http://127.0.0.1:11434", model="test", data_dir=tmp_path,
                  allowed_hosts=["testserver", "localhost", "127.0.0.1"])


async def eventually(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.02)
