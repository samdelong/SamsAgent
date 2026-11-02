import asyncio
import json
import shlex
import sys

from samsagent.runtime import Runtime
from conftest import FakeModel, eventually, response


async def test_end_to_end_research_experiment_and_independent_review(config):
    """Deterministic model, real MCP and Python experiment; no live research claims."""
    stage = {"coordinator": 0, "reviewer": 0}
    report_path = config.data_dir / "report.md"
    evidence_path = config.data_dir / "experiment.txt"
    job_id = None

    def model(messages, tools):
        nonlocal job_id
        reviewer = "(reviewer)" in messages[0]["content"]
        role = "reviewer" if reviewer else "coordinator"
        step = stage[role]
        stage[role] += 1
        if reviewer:
            if step == 0:
                return response(calls=[("machine__file_read", {"path": str(report_path)}),
                                       ("machine__file_read", {"path": str(evidence_path)})])
            tool_results = [m["content"] for m in messages if m["role"] == "tool"]
            assert any("1.0" in r for r in tool_results)
            return response(calls=[("finish_review", {"approved": True, "findings": "Verified report cites the source and the executed experiment."})])
        if step == 0:
            return response(calls=[("discover_tools", {"query": "file process"})])
        if step == 1:
            script = "import math; xs=[1e16,1.,-1e16]; total=0.\nfor x in xs: total+=x\nprint(total, math.fsum(xs))"
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} > {shlex.quote(str(evidence_path))}"
            return response(calls=[("machine__process_start", {"command": command, "cwd": str(config.data_dir)})])
        if step == 2:
            result = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
            job = result.get("structuredContent") or json.loads(result["content"][0]["text"])
            job_id = job["id"]
            return response(calls=[("machine__process_status", {"job_id": job_id})])
        if step == 3:
            return response(calls=[("machine__file_read", {"path": str(evidence_path)})])
        if step == 4:
            return response(calls=[("machine__file_write", {"path": str(report_path), "content":
                "# Summation experiment\n\nSource: https://docs.python.org/3/library/math.html#math.fsum\n"
                "This controlled fixture tests [1e16, 1., -1e16]. Sequential += yields 0.0; math.fsum yields 1.0.\n"
                f"Evidence: {evidence_path}\nLimit: one sequence, not a broad numerical benchmark."})])
        return response(calls=[("propose_completion", {"report": "Experiment and cited report ready", "artifacts": [str(report_path), str(evidence_path)]})])

    runtime = Runtime(config, FakeModel(model))
    await runtime.start()
    try:
        session = runtime.create_session("Compare sequential floating-point addition and math.fsum", "Produce a reproducible experiment and cited report", "goal")
        await eventually(lambda: runtime.store.one("sessions", session["id"])["status"] in {"completed", "blocked"}, 20)
        data = runtime.store.one("sessions", session["id"])
        assert data["status"] == "completed", runtime.store.rows("SELECT role,status,result FROM agents")
        assert evidence_path.read_text().strip() == "0.0 1.0"
        assert "https://docs.python.org" in report_path.read_text()
        assert runtime.store.rows("SELECT id FROM agents WHERE role='reviewer' AND status='completed'")
    finally:
        await runtime.close()
