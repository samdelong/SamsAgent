import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from .config import Config
from .runtime import Runtime
from .workflows import create_workflow


class NewSession(BaseModel):
    goal: str = Field(min_length=1, max_length=12000)
    criteria: str = Field(default="", max_length=8000)
    mode: Literal["chat", "goal"] = "chat"
    token_budget: int | None = Field(default=None, gt=0)
    time_budget: float | None = Field(default=None, gt=0)


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=32000)
    agent_id: str | None = None


class Control(BaseModel):
    action: Literal["pause", "resume", "cancel"]


class Workflow(BaseModel):
    agent_id: str
    path: str


def create_app(config, runtime=None):
    runtime = runtime or Runtime(config)

    @asynccontextmanager
    async def lifespan(app):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="SamsAgent", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.allowed_hosts)

    @app.middleware("http")
    async def same_origin(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if (request.headers.get("sec-fetch-site") == "cross-site" or
                (origin and urlsplit(origin).netloc != request.headers.get("host"))):
                return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return JSONResponse({"detail": "Use application/json"}, status_code=415)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; object-src 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "model": config.model, "context_tokens": config.context_tokens}

    @app.get("/api/sessions")
    async def sessions():
        return runtime.store.rows("SELECT * FROM sessions ORDER BY created DESC")

    @app.post("/api/sessions")
    async def new_session(body: NewSession):
        return runtime.create_session(**body.model_dump())

    @app.get("/api/sessions/{session_id}")
    async def session(session_id: str):
        data = runtime.store.one("sessions", session_id)
        data["agents"] = runtime.store.rows("SELECT * FROM agents WHERE session_id=?", (session_id,))
        data["workflows"] = runtime.store.rows("SELECT * FROM workflows WHERE session_id=?", (session_id,))
        directory = runtime.store.session_dir(session_id)
        data["state"] = (directory / "STATE.md").read_text()
        data["artifacts"] = [str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and not p.name.startswith("inference-")]
        return data

    @app.post("/api/sessions/{session_id}/messages")
    async def message(session_id: str, body: Message):
        runtime.store.one("sessions", session_id)
        agent = body.agent_id or runtime.store.rows("SELECT id FROM agents WHERE session_id=? AND role='coordinator'", (session_id,))[0]["id"]
        if runtime.store.one("agents", agent)["session_id"] != session_id:
            raise HTTPException(404)
        runtime.send(agent, body.text)
        return {"queued": True}

    @app.post("/api/sessions/{session_id}/control")
    async def control(session_id: str, body: Control):
        await runtime.control(session_id, body.action)
        return {"ok": True}

    @app.get("/api/agents/{agent_id}")
    async def agent(agent_id: str, after: int = 0):
        data = runtime.store.one("agents", agent_id)
        data["messages"] = [{"id": n, **m} for n, m in runtime.store.history(agent_id, after)]
        return data

    @app.post("/api/workflows")
    async def workflow(body: Workflow):
        agent = runtime.store.one("agents", body.agent_id)
        runtime.check_active(agent)
        return {"id": create_workflow(runtime, agent, Path(body.path))}

    @app.get("/api/workflows/{workflow_id}")
    async def workflow_status(workflow_id: str):
        return runtime.store.one("workflows", workflow_id)

    @app.get("/api/sessions/{session_id}/events")
    async def events(session_id: str, request: Request, after: int = 0):
        runtime.store.one("sessions", session_id)
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            raise HTTPException(400, "Invalid event cursor")

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                rows = runtime.store.rows("SELECT * FROM events WHERE session_id=? AND id>? ORDER BY id LIMIT 100", (session_id, cursor))
                for event in rows:
                    event["data"] = json.loads(event["data"])
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                if not rows:
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/sessions/{session_id}/artifacts/{path:path}")
    async def artifact(session_id: str, path: str):
        runtime.store.one("sessions", session_id)
        root = runtime.store.session_dir(session_id).resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target, filename=target.name, media_type="application/octet-stream")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    config = Config.load(args.config)
    uvicorn.run(create_app(config), host="127.0.0.1", port=config.port, workers=1)


if __name__ == "__main__":
    main()
