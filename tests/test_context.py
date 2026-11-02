import json

import pytest

from samsagent.runtime import Runtime
from samsagent.context import ContextOverflow
from conftest import EmptyMCP, FakeModel, response


async def test_repeated_compaction_preserves_evidence_and_complete_tool_pairs(config):
    model = FakeModel(lambda m, t: response("Goal: compare results. Evidence: /evidence/results.csv. Pending child: abc. Next: verify."))
    runtime = Runtime(config, model, EmptyMCP())
    session = runtime.create_session("compare", mode="goal")
    try:
        for round_number in range(2):
            for i in range(18):
                runtime.store.message(session["agent_id"], {"role": "user", "content": f"step {i} " + "x" * 1600})
                runtime.store.message(session["agent_id"], response(calls=[("file_read", {"path": "/evidence/results.csv"})])["message"])
                runtime.store.message(session["agent_id"], {"role": "tool", "tool_name": "file_read", "content": "y" * 1600})
            agent = runtime.store.one("agents", session["agent_id"])
            context = await runtime.context.build(agent, "Original goal. Active child abc.", [])
            updated = runtime.store.one("agents", session["agent_id"])
            assert updated["compacted_through"] > agent["compacted_through"]
            assert "/evidence/results.csv" in updated["summary"]
            assert "Active child abc" in context[0]["content"]
            for i, message in enumerate(context):
                if message["role"] == "tool":
                    assert context[i-1].get("tool_calls")
            assert runtime.context.estimate(context) + config.output_tokens < config.context_tokens
        assert len(list((runtime.store.agent_dir(session["agent_id"]) / "checkpoints").glob("*.md"))) == 2
        assert len(runtime.store.history(session["agent_id"])) == 109
    finally:
        await runtime.close()


async def test_failed_compaction_keeps_prior_checkpoint_and_bounds_retries(config):
    count = 0

    def fail(messages, tools):
        nonlocal count
        count += 1
        raise ValueError("broken summary")

    runtime = Runtime(config, FakeModel(fail), EmptyMCP())
    session = runtime.create_session("goal")
    try:
        for _ in range(20):
            runtime.store.message(session["agent_id"], {"role": "user", "content": "x" * 3000})
        runtime.store.execute("UPDATE agents SET summary='existing evidence' WHERE id=?", (session["agent_id"],))
        agent = runtime.store.one("agents", session["agent_id"])
        with pytest.raises(ContextOverflow):
            await runtime.context.build(agent, "system", [])
        assert runtime.store.one("agents", agent["id"])["summary"] == "existing evidence"
        assert runtime.store.one("agents", agent["id"])["compacted_through"] == 0
        assert count < 10
    finally:
        await runtime.close()


async def test_compaction_retries_smaller_chunks(config):
    sizes = []

    def summarize(messages, tools):
        size = len(messages[-1]["content"])
        sizes.append(size)
        if size > 2500:
            raise ValueError("too much input")
        return response("Preserved checkpoint")

    runtime = Runtime(config, FakeModel(summarize), EmptyMCP())
    session = runtime.create_session("goal")
    try:
        result = await runtime.context.summarize(runtime.store.one("agents", session["agent_id"]), "", ["z" * 12000])
        assert result == "Preserved checkpoint"
        assert max(sizes) > 2500 and min(sizes) < 2500
    finally:
        await runtime.close()


async def test_pause_during_compaction_preserves_checkpoint(config):
    from samsagent.runtime import Suspended

    def pause(messages, tools):
        runtime.store.execute("UPDATE sessions SET status='paused'")
        raise Suspended()

    runtime = Runtime(config, FakeModel(pause), EmptyMCP())
    session = runtime.create_session("goal")
    try:
        for _ in range(20):
            runtime.store.message(session["agent_id"], {"role": "user", "content": "x" * 3000})
        agent = runtime.store.one("agents", session["agent_id"])
        with pytest.raises(Suspended):
            await runtime.context.build(agent, "system", [])
        assert runtime.store.one("agents", agent["id"])["compacted_through"] == 0
        assert runtime.store.one("sessions", session["id"])["status"] == "paused"
        assert len(runtime.model.calls) == 1
    finally:
        await runtime.close()
