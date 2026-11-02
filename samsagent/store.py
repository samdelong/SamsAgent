import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid


def uid():
    return uuid.uuid4().hex


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "state.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          PRAGMA journal_mode=WAL;
          PRAGMA foreign_keys=ON;
          CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, goal TEXT NOT NULL, criteria TEXT NOT NULL,
            mode TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
            token_budget INTEGER, time_budget REAL, tokens INTEGER NOT NULL DEFAULT 0,
            active_seconds REAL NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
            parent_id TEXT, role TEXT NOT NULL, task TEXT NOT NULL, depth INTEGER NOT NULL,
            status TEXT NOT NULL, result TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '', compacted_through INTEGER NOT NULL DEFAULT 0,
            selected TEXT NOT NULL DEFAULT '[]', stalls INTEGER NOT NULL DEFAULT 0,
            recovering INTEGER NOT NULL DEFAULT 0, progress TEXT NOT NULL DEFAULT '');
          CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL REFERENCES agents(id),
            body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL REFERENCES agents(id),
            text TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
            kind TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS operations (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, name TEXT NOT NULL,
            args TEXT NOT NULL, status TEXT NOT NULL, result TEXT, created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, session_id TEXT NOT NULL,
            status TEXT NOT NULL, source TEXT NOT NULL, checkpoint TEXT NOT NULL DEFAULT '{}',
            result TEXT NOT NULL DEFAULT '');
          CREATE TABLE IF NOT EXISTS evidence (
            session_id TEXT NOT NULL, digest TEXT NOT NULL,
            PRIMARY KEY(session_id,digest));
          CREATE INDEX IF NOT EXISTS messages_agent ON messages(agent_id,id);
          CREATE INDEX IF NOT EXISTS events_session ON events(session_id,id);
          CREATE INDEX IF NOT EXISTS operations_agent ON operations(agent_id,status);
          CREATE INDEX IF NOT EXISTS agents_session ON agents(session_id,status);
        """)
        self.db.commit()

    def execute(self, sql, args=()):
        result = self.db.execute(sql, args)
        self.db.commit()
        return result

    def rows(self, sql, args=()):
        return [dict(row) for row in self.db.execute(sql, args).fetchall()]

    def one(self, table, identifier):
        if table not in {"sessions", "agents", "operations", "workflows"}:
            raise ValueError("Unknown table")
        row = self.db.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return dict(row)

    def event(self, session, kind, data):
        return self.execute(
            "INSERT INTO events(session_id,kind,data,created) VALUES(?,?,?,?)",
            (session, kind, json.dumps(data, ensure_ascii=False), time.time()),
        ).lastrowid

    def message(self, agent, body):
        return self.execute("INSERT INTO messages(agent_id,body) VALUES(?,?)",
                            (agent, json.dumps(body, ensure_ascii=False))).lastrowid

    def history(self, agent, after=0):
        return [(r["id"], json.loads(r["body"])) for r in self.rows(
            "SELECT * FROM messages WHERE agent_id=? AND id>? ORDER BY id", (agent, after))]

    def drain_inbox(self, agent):
        with self.db:
            rows = self.rows("SELECT * FROM inbox WHERE agent_id=? ORDER BY id", (agent,))
            for row in rows:
                self.db.execute("INSERT INTO messages(agent_id,body) VALUES(?,?)",
                                (agent, json.dumps({"role": "user", "content": row["text"]})))
                self.db.execute("DELETE FROM inbox WHERE id=?", (row["id"],))

    def session_dir(self, session):
        return self.root / "sessions" / session

    def agent_dir(self, agent):
        row = self.one("agents", agent)
        return self.session_dir(row["session_id"]) / "agents" / agent

    def artifact(self, session, text, prefix="output"):
        path = self.session_dir(session) / "artifacts" / f"{prefix}-{uid()}.txt"
        atomic_write(path, text)
        return str(path)

    def close(self):
        self.db.close()

