def tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or [], "additionalProperties": False}}}


S = {"type": "string"}
TOOLS = [
    tool("discover_tools", "Find MCP tools by description; selected schemas become available next turn.",
         {"query": S}, ["query"]),
    tool("spawn", "Delegate a focused task; references should be file paths or URLs, not full transcripts.",
         {"task": S, "references": {"type": "array", "items": S}}, ["task"]),
    tool("agent_message", "Send a follow-up to an agent in this session.", {"agent_id": S, "text": S}, ["agent_id", "text"]),
    tool("inspect_agents", "List agents and workflows in this session."),
    tool("wait", "Wait up to 30 seconds for an agent; releases execution capacity.", {"agent_id": S}, ["agent_id"]),
    tool("cancel_agent", "Cancel an agent and its descendants.", {"agent_id": S}, ["agent_id"]),
    tool("write_notes", "Save your durable Markdown notes. Only the coordinator can update shared STATE.md.",
         {"content": S, "shared": {"type": "boolean"}}, ["content"]),
    tool("start_workflow", "Run a saved Python file implementing async def run(ctx). Use discover_runtime for API help.",
         {"path": S}, ["path"]),
    tool("discover_runtime", "Read the Python workflow API and checkpoint rules."),
    tool("inspect_workflow", "Read a workflow's status, checkpoint, result and source path.", {"workflow_id": S}, ["workflow_id"]),
    tool("resolve_operation", "After inspecting effects, resolve an uncertain operation; record evidence and whether retry is safe.",
         {"operation_id": S, "evidence": S, "retry_safe": {"type": "boolean"}},
         ["operation_id", "evidence", "retry_safe"]),
    tool("propose_completion", "Request independent review of deliverables against the goal; does not itself complete the goal.",
         {"report": S, "artifacts": {"type": "array", "items": S, "minItems": 1}}, ["report", "artifacts"]),
    tool("finish_review", "Reviewer only: give a verdict after inspecting evidence. List concrete deficiencies if rejecting.",
         {"approved": {"type": "boolean"}, "findings": S}, ["approved", "findings"]),
    tool("block", "Record a genuine blocker that requires input or external change.", {"reason": S}, ["reason"]),
]
SCHEMAS = {t["function"]["name"]: t for t in TOOLS}

WORKFLOW_HELP = '''Write a Python file with async def run(ctx). The runtime calls it in a child process.
await ctx.spawn(task, references=[]) -> agent id
await ctx.wait(agent_id) -> {id,status,result}; returns after at most 30 seconds; loop if needed
await ctx.message(agent_id, text)
await ctx.llm(prompt) -> assistant text (shares inference limits/accounting)
await ctx.tool(name, arguments) -> MCP result; use names returned by discover_tools
await ctx.checkpoint() -> saved JSON object
await ctx.checkpoint({"phase": "...", "child_id": "..."}) -> atomically save JSON object
On restart run(ctx) starts at the beginning. Read your checkpoint and skip finished work.
Save spawned agent IDs immediately and reuse them after restart. Use asyncio.gather for parallel work.
Direct Python side effects are not journaled. Prefer ctx.tool for tracked operations. A crash between
an operation and your checkpoint can leave an uncertain result; inspect effects before repeating it.
The runtime snapshots your source. Return a JSON-serializable result. Print goes to a log file.
'''
