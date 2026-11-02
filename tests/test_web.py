import asyncio
import json
from pathlib import Path

import httpx
import pytest

from samsagent.runtime import Runtime
from samsagent.web import create_app
from conftest import EmptyMCP, FakeModel, eventually


async def test_web_controls_origin_validation_and_artifact_escape(config, tmp_path):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    app = create_app(config, runtime)
    await runtime.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://testserver") as client:
            assert (await client.get("/")).status_code == 200
            assert "textContent" in (await client.get("/static/app.js")).text
            assert (await client.post("/api/sessions", json={"goal": "bad"}, headers={"Origin": "https://attacker.example"})).status_code == 403
            assert (await client.post("/api/sessions", content="{}", headers={"Content-Type": "text/plain"})).status_code == 415
            assert (await client.get("/api/sessions", headers={"Host": "evil.example"})).status_code == 400
            assert (await client.post("/api/sessions", json={"goal": "", "mode": "invalid"})).status_code == 422
            session = (await client.post("/api/sessions", json={"goal": "<script>alert(1)</script>"})).json()
            await eventually(lambda: runtime.store.one("agents", session["agent_id"])["status"] == "idle")
            data = (await client.get(f"/api/sessions/{session['id']}")).json()
            assert data["goal"] == "<script>alert(1)</script>"
            for action in ["pause", "resume"]:
                assert (await client.post(f"/api/sessions/{session['id']}/control", json={"action": action})).status_code == 200
            assert (await client.post(f"/api/sessions/{session['id']}/messages", json={"text": "new direction"})).status_code == 200
            directory = runtime.store.session_dir(session["id"])
            secret = tmp_path / "outside.txt"
            secret.write_text("not an artifact")
            (directory / "escape.txt").symlink_to(secret)
            assert (await client.get(f"/api/sessions/{session['id']}/artifacts/escape.txt")).status_code == 404
            download = await client.get(f"/api/sessions/{session['id']}/artifacts/GOAL.md")
            assert download.status_code == 200
            assert "attachment" in download.headers["content-disposition"]
            assert "default-src 'self'" in download.headers["content-security-policy"]
            assert (await client.get("/api/sessions/missing")).status_code == 404
    finally:
        await runtime.close()


async def test_event_stream_reconnect_uses_persisted_cursor(config):
    runtime = Runtime(config, FakeModel(), EmptyMCP())
    session = runtime.create_session("events")
    app = create_app(config, runtime)
    route = next(r for r in app.routes if getattr(r, "path", "").endswith("/events"))

    class Request:
        headers = {"last-event-id": "1"}

        async def is_disconnected(self):
            return False

    try:
        runtime.store.event(session["id"], "second", {"value": 2})
        stream = await route.endpoint(session["id"], Request(), 0)
        first = await anext(stream.body_iterator)
        identifier = int(first.splitlines()[0].split(": ")[1])
        assert identifier > 1
        await stream.body_iterator.aclose()
        Request.headers = {"last-event-id": str(identifier)}
        stream = await route.endpoint(session["id"], Request(), 0)
        second = await anext(stream.body_iterator)
        assert "second" in second and f"id: {identifier}\n" not in second
        await stream.body_iterator.aclose()
    finally:
        await runtime.close()
