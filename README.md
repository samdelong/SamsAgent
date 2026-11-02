# SamsAgent

A persistent, single-user research agent for a Linux machine you control. Connect an existing **tool-capable Ollama model**, open the web interface, and give it a research question. It can use MCP tools, run experiments, delegate to workers, and create Python workflows.

Goal mode keeps working until a separate reviewer accepts the deliverables or the agent records a blocker. Normal chat responds and waits. Both retain conversations and Markdown research state across restarts.

## Run locally

Requires Python 3.12+, Linux for deployment, and a reachable Ollama server. Development and most tests also work on macOS.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -c requirements.lock -e '.[test]'
cp config.example.toml config.toml
```

Edit `config.toml`: set `ollama_url` to the endpoint reachable **from the harness machine**, and `model` to an installed Ollama model tag. No model is downloaded automatically. Use native tool calling; unsupported models fail startup rather than falling back to parsing prose as commands.

```sh
.venv/bin/samsagent --config config.toml
```

Open http://127.0.0.1:8765. The interface supports goals and acceptance criteria, optional token/time budgets, follow-ups, agent conversations, workflow status, notes, artifacts, and pause/resume/cancel. Follow-ups target the coordinator; the API can also target individual workers.

Defaults: 16,384-token context, 2,048-token output allowance, one simultaneous inference request, eight execution slots, and three delegation levels. Increase context/concurrency after measuring memory use and latency on your Ollama server. A model's advertised context does not guarantee the configured window fits in VRAM.

## EC2 and Tailscale

Install the checkout and virtual environment at `/opt/samsagent`, owned by the service account. Store `data_dir` on persistent EBS storage and use an absolute path in production. Keep `config.toml` readable only by the service account if it contains MCP credentials.

1. Install and authenticate Tailscale on EC2 and your client device.
2. Add the machine's exact `*.ts.net` hostname to `allowed_hosts` in `config.toml`.
3. Adjust `User`, `WorkingDirectory`, and `ExecStart` in `deploy/samsagent.service` if needed, then install it:

```sh
sudo cp deploy/samsagent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now samsagent
sudo tailscale serve --bg http://127.0.0.1:8765
```

Use the private HTTPS URL printed by Tailscale. Tailscale Serve and your tailnet ACLs provide access control; there is no application account system. Restrict access to your devices/users. Keep port 8765 and Ollama off the public internet; the app binds only to loopback. Use Serve, not Funnel. The Ollama URL may also use a private Tailscale address.

```sh
journalctl -u samsagent -f
```

The unit restarts on failure. `KillMode=process` intentionally lets detached experiments continue across harness restarts. **Cancel in the UI** terminates tracked commands and workflows; stopping the service does not stop detached experiments.

## Research and tools

The bundled `machine` MCP server provides file read/write/search and shell start/status/output/terminate. Commands use `/bin/sh`, an explicit working directory, and the service account's permissions. There is no approval layer or filesystem sandbox. Agents can install utilities or use `curl`/Python to retrieve sources. Configure an additional browser/search MCP server for richer web research; no commercial search service or browser dependency is included.

External MCP servers support stdio and Streamable HTTP; see `config.example.toml`. Tool names are `server__tool`. `discover_tools` searches descriptions and selects up to eight schemas for subsequent requests. Configure external environment variables and HTTP headers explicitly; SDK stdio environment inheritance is minimal.

Agents can spawn, message, inspect, wait for, and cancel workers. Waiting holds no inference or execution slot. Model calls from workers, workflows, reviewers, and compression share concurrency limits and accounting. Agents choose direct work or delegation. A coordinator's completion claim must pass a fresh-context reviewer that has inspected evidence.

```text
data/
  state.sqlite3                 execution journal, inboxes, events, checkpoints
  processes/                    shell metadata and output logs
  sessions/<session-id>/
    GOAL.md                     original goal and acceptance criteria
    STATE.md                    coordinator's findings and next actions
    agents/<agent-id>/
      NOTES.md                  agent-owned working notes
      checkpoints/*.md         versioned context summaries
    artifacts/                  full tool outputs and inference records
    workflows/<workflow-id>/
      source.py                 immutable source snapshot for this run
      stderr.log                workflow prints and errors
```

Store deliverables inside the session directory to make them downloadable in the UI. SQLite is authoritative for scheduling; Markdown is durable research memory. The runtime atomically replaces notes. Shell access can still change files directly: ownership rules are coordination conventions, not an OS security boundary.

## Context and recovery

The runtime budgets instructions, tool schemas, notes, recent complete exchanges, and output reserve before inference. Compaction begins near 70% occupancy and targets a short summary plus recent history below 40%. Estimates start conservatively and calibrate upward using Ollama's reported usage; they are not an exact tokenizer. Oversized pinned context stops with an error rather than deliberately dropping the goal.

Compression preserves evidence paths, decisions, pending work, and next actions. Full transcripts remain on disk. Large tool results become bounded previews with retrievable paths. Failed summaries retry smaller chunks without replacing a valid checkpoint. Keep pinned notes concise and save large sources separately.

Operations are journaled before execution. After restart, unfinished operations become **uncertain**. Identical calls are blocked until `resolve_operation` records inspection evidence and says retry is safe. Interrupted tool exchanges receive a recovery result. Operations resolved as unsafe to repeat remain blocked from identical re-execution.

Pause lets active operations settle and prevents subsequent work. Resume continues saved state. Cancel stops scheduling and attempts to terminate tracked process groups; untracked processes created by arbitrary code cannot be guaranteed to stop. A blocked goal resumes with new information or Resume. Time budgets count active session time, excluding paused/offline time; token budgets include all agents' input and output tokens. An exhausted budget pauses work; Resume does not bypass it.

Progress is a heuristic: new tool evidence, changed notes, or completed agents count; repeated identical results do not. Three empty goal cycles prompt a changed approach; a stalled recovery blocks. Tool-only loops get twelve turns before that check. This catches mechanical repetition, not every form of unproductive reasoning.

## Python workflows

Write `async def run(ctx)` in a Python file and ask the agent to `start_workflow` with its absolute path. `examples/research_workflow.py` demonstrates parallel source research and an experiment followed by synthesis.

```python
async def run(ctx):
    state = await ctx.checkpoint()
    if "worker" not in state:
        state["worker"] = await ctx.spawn("Run the experiment and save its results")
        await ctx.checkpoint(state)
    while True:
        result = await ctx.wait(state["worker"])
        if result["status"] in {"completed", "failed", "blocked", "cancelled"}:
            return result
```

| Call | Behavior |
| --- | --- |
| `ctx.spawn(task, references=[])` | Create a worker; return its ID. |
| `ctx.wait(agent_id)` | Wait up to 30 seconds; return ID, status, result. |
| `ctx.message(agent_id, text)` | Queue a follow-up. |
| `ctx.llm(prompt)` | Make a metered model call; return text. |
| `ctx.tool(name, arguments)` | Call a namespaced MCP tool through the journal. |
| `ctx.checkpoint()` | Read the saved JSON object. |
| `ctx.checkpoint(state)` | Atomically replace that object. |

Use Python loops, conditions, and `asyncio.gather` to build graphs. Source is snapshotted and executed in a child process with private JSON-lines pipes. Prints go to `stderr.log`. Interrupted workflows restart at their entry point using the saved checkpoint. They do **not** resume a Python stack or guarantee exactly-once effects. A crash between a completed operation and its next checkpoint may need reconciliation. Prefer SDK tools over untracked effects. Pause gates SDK calls; arbitrary CPU work in generated Python can continue until cancellation.

## HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Model/configuration summary. |
| `GET/POST /api/sessions` | List sessions or submit `goal`, `criteria`, `mode`, optional budgets. |
| `GET /api/sessions/{id}` | Goal, agents, workflows, notes, artifacts. |
| `POST /api/sessions/{id}/messages` | Submit `text` and optional `agent_id`. |
| `POST /api/sessions/{id}/control` | `action`: `pause`, `resume`, or `cancel`. |
| `GET /api/agents/{id}?after=N` | Agent and messages after a message ID. |
| `POST /api/workflows` | Start a saved Python `path` for an `agent_id`. |
| `GET /api/workflows/{id}` | Status, source, checkpoint, result. |
| `GET /api/sessions/{id}/events` | SSE; reconnect with `Last-Event-ID` or `?after=N`. |
| `GET /api/sessions/{id}/artifacts/{path}` | Download a file inside this session. |

Writes require JSON and reject cross-origin browser requests. Artifact paths cannot traverse outside a session or follow escaping symlinks. Model output is rendered as text rather than executable HTML.

## Tests and limits

```sh
.venv/bin/python -m pytest -q
```

Tests use deterministic models, real MCP stdio/HTTP transports, actual shell experiments, workflow subprocesses, compression, reviewer rejection/approval, budget/pause/cancel handling, SSE cursors, origin/path protections, and crash recovery. They test the harness, not your 27B model's research ability. The end-to-end test is a controlled fixture, not a live web-research evaluation.

The actual Ollama endpoint, EC2 machine, systemd unit, and Tailscale deployment need a live smoke test in your environment. No endpoint or EC2 credentials are assumed. Linux process identity recovery uses `/proc`; after a server restart macOS may report a surviving process as uncertain rather than signal an unverified PID.

This is one process on one machine. No visual graph editor, vector database, distributed execution, automatic artifact retention, or multi-user isolation is included. Transcripts/logs accumulate: size storage accordingly and back it up. Stop the service or use SQLite's backup API rather than copying a live database without its WAL. Models can omit evidence during compression or make mistaken reviews; inspect artifacts for important conclusions.
