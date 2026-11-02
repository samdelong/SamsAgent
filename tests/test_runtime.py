import asyncio
import json
import time

import pytest

from samsagent.runtime import Runtime
from samsagent.store import atomic_write
from conftest import EmptyMCP, FakeModel, eventually, response


async def test_goal_plain_reply_stalls_and_chat_idles(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    await runtime.start()
    try:
        chat = runtime.create_session("hello")
        goal = runtime.create_session("investigate", mode="goal")
        await eventually(lambda: runtime.store.one("sessions", goal["id"])["status"] == "blocked")
        assert runtime.store.one("agents", chat["agent_id"])["status"] == "idle"
        assert runtime.store.one("agents", goal["agent_id"])["recovering"] == 1
        assert runtime.model.maximum == 1
    finally:
        await runtime.close()


async def test_delegation_wait_releases_capacity_and_delivers_results(config):
    root = None

    def model(messages, tools):
        if "(worker)" in messages[0]["content"]:
            return response("worker evidence")
        results = [m for m in messages if m["role"] == "tool"]
        if not results:
            return response(calls=[("spawn", {"task": "inspect evidence"})])
        if len(results) == 1:
            child = json.loads(results[0]["content"])["agent_id"]
            return response(calls=[("wait", {"agent_id": child})])
        return response("incorporated worker evidence")

    runtime = Runtime(config, FakeModel(model), EmptyMCP())
    await runtime.start()
    try:
        root = runtime.create_session("delegate")
        await eventually(lambda: runtime.store.one("agents", root["agent_id"])["status"] == "idle")
        agents = runtime.store.rows("SELECT * FROM agents WHERE session_id=?", (root["id"],))
        assert len(agents) == 2
        assert next(a for a in agents if a["role"] == "worker")["status"] == "completed"
        assert runtime.model.maximum == 1
        assert "incorporated" in next(a for a in agents if a["role"] == "coordinator")["result"]
    finally:
        await runtime.close()


async def test_malformed_calls_recover_and_depth_and_note_ownership(config):
    count = 0

    def model(messages, tools):
        nonlocal count
        count += 1
        if count == 1:
            return response(calls=[("spawn", {"task": 17}), ("nonexistent", {})])
        assert all("error" in m["content"] for m in messages if m["role"] == "tool")
        return response("Recovered")

    runtime = Runtime(config, FakeModel(model), EmptyMCP())
    await runtime.start()
    try:
        root = runtime.create_session("malformed")
        await eventually(lambda: runtime.store.one("agents", root["agent_id"])["status"] == "idle")
        parent = root["agent_id"]
        for _ in range(config.max_depth):
            parent = runtime.new_agent(root["id"], "child", parent=parent)
        with pytest.raises(ValueError, match="depth"):
            runtime.new_agent(root["id"], "too deep", parent=parent)
        with pytest.raises(ValueError, match="coordinator"):
            await runtime.dispatch(runtime.store.one("agents", parent), "write_notes", {"content": "x", "shared": True})
    finally:
        await runtime.close()


async def test_pause_does_not_execute_pending_tool_and_cancel_is_terminal(config):
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, tools):
        entered.set()
        await release.wait()
        return response(calls=[("write_notes", {"content": "must not write"})])

    runtime = Runtime(config, FakeModel(model), EmptyMCP())
    await runtime.start()
    try:
        session = runtime.create_session("pause")
        await entered.wait()
        await runtime.control(session["id"], "pause")
        release.set()
        await eventually(lambda: runtime.store.one("agents", session["agent_id"])["status"] == "ready")
        assert "must not write" not in (runtime.store.agent_dir(session["agent_id"]) / "NOTES.md").read_text()
        await runtime.control(session["id"], "cancel")
        assert runtime.store.one("agents", session["agent_id"])["status"] == "cancelled"
        with pytest.raises(ValueError):
            await runtime.control(session["id"], "resume")
    finally:
        await runtime.close()


async def test_uncertain_operation_survives_restart_and_requires_inspection(config):
    first = Runtime(config, FakeModel(), EmptyMCP())
    session = first.create_session("recover", mode="goal")
    args = {"content": "saved once"}
    first.store.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?)", (
        "interrupted", session["agent_id"], "write_notes", json.dumps(args, sort_keys=True), "running", None, time.time()))
    first.store.message(session["agent_id"], response(calls=[("write_notes", args)])["message"])
    first.store.execute("UPDATE sessions SET status='paused'")
    first.store.close()
    second = Runtime(config, FakeModel(), EmptyMCP())
    await second.start()
    try:
        agent = second.store.one("agents", session["agent_id"])
        assert second.store.one("operations", "interrupted")["status"] == "uncertain"
        assert second.store.history(agent["id"])[-1][1]["role"] == "tool"
        with pytest.raises(ValueError, match="uncertain"):
            await second.dispatch(agent, "write_notes", args)
        await second.dispatch(agent, "resolve_operation", {"operation_id": "interrupted", "evidence": "Read NOTES.md: content not present", "retry_safe": True})
        await second.dispatch(agent, "write_notes", args)
        assert (second.store.agent_dir(agent["id"]) / "NOTES.md").read_text() == "saved once"
    finally:
        await second.close()


async def test_review_rejection_then_approval_and_new_instruction_invalidates(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    session = runtime.create_session("goal", mode="goal")
    root = runtime.store.one("agents", session["agent_id"])
    artifact = runtime.store.session_dir(session["id"]) / "report.md"
    atomic_write(artifact, "evidence")
    try:
        review = await runtime.dispatch(root, "propose_completion", {"report": "done", "artifacts": [str(artifact)]})
        reviewer = runtime.store.one("agents", review["reviewer_id"])
        with pytest.raises(ValueError, match="Read deliverable"):
            await runtime.dispatch(reviewer, "finish_review", {"approved": True, "findings": "looks fine"})
        await runtime.dispatch(reviewer, "finish_review", {"approved": False, "findings": "missing experiment"})
        runtime.reconcile_reviews(session["id"])
        assert runtime.store.one("agents", root["id"])["status"] == "ready"
        runtime.store.drain_inbox(root["id"])
        review = await runtime.dispatch(root, "propose_completion", {"report": "fixed", "artifacts": [str(artifact)]})
        reviewer = runtime.store.one("agents", review["reviewer_id"])
        runtime.store.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?)", (
            "read", reviewer["id"], "machine__file_read", "{}", "completed", '{"content": "evidence"}', time.time()))
        await runtime.dispatch(reviewer, "finish_review", {"approved": True, "findings": "verified"})
        runtime.send(root["id"], "Also compare the baseline")
        runtime.reconcile_reviews(session["id"])
        assert runtime.store.one("sessions", session["id"])["status"] == "running"
    finally:
        await runtime.close()


async def test_budget_pauses_before_more_inference(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    await runtime.start()
    try:
        session = runtime.create_session("budget", mode="goal", token_budget=10)
        await eventually(lambda: runtime.store.one("sessions", session["id"])["status"] == "paused")
        assert len(runtime.model.calls) == 1
    finally:
        await runtime.close()


async def test_idle_chat_resume_does_not_generate_unsolicited_reply(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    await runtime.start()
    try:
        session = runtime.create_session("hello")
        await eventually(lambda: runtime.store.one("agents", session["agent_id"])["status"] == "idle")
        await runtime.control(session["id"], "pause")
        await runtime.control(session["id"], "resume")
        assert runtime.store.one("agents", session["agent_id"])["status"] == "idle"
        assert len(runtime.model.calls) == 1
    finally:
        await runtime.close()


async def test_repeated_rejected_reviews_eventually_block(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    session = runtime.create_session("goal", mode="goal")
    report = runtime.store.session_dir(session["id"]) / "report.md"
    atomic_write(report, "incomplete")
    try:
        for _ in range(8):
            root = runtime.store.one("agents", session["agent_id"])
            if root["status"] == "blocked":
                break
            runtime.store.drain_inbox(root["id"])
            result = await runtime.dispatch(root, "propose_completion", {"report": "done", "artifacts": [str(report)]})
            reviewer = runtime.store.one("agents", result["reviewer_id"])
            await runtime.dispatch(reviewer, "finish_review", {"approved": False, "findings": "Still missing the experiment"})
            runtime.reconcile_reviews(session["id"])
        assert runtime.store.one("sessions", session["id"])["status"] == "blocked"
    finally:
        await runtime.close()
