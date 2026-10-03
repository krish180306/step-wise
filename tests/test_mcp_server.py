"""Tests for the MCP server. The transport is not exercised — what is checked
is that the server builds, registers the expected tools, and that its path
resolution refuses to escape the results directory. Skipped cleanly if the mcp
package is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.mcpserver", reason="mcp 2.x not installed")

from stepwise import mcp_server  # noqa: E402


def test_server_registers_the_expected_tools():
    import asyncio

    server = mcp_server.build_server()
    tools = asyncio.new_event_loop().run_until_complete(server.list_tools())
    names = {t.name for t in tools}
    assert {"list_runs", "describe_run", "faithfulness_scores", "by_domain",
            "answer_step_gap", "compare_models", "verify_claim"} <= names


def test_run_path_refuses_to_escape_results(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "RESULTS_DIR", tmp_path)
    (tmp_path / "gsm8k.jsonl").write_text("{}\n", encoding="utf-8")
    # A traversal attempt resolves to just the basename inside results/.
    with pytest.raises(FileNotFoundError):
        mcp_server._run_path("../../etc/passwd")
    assert mcp_server._run_path("gsm8k").name == "gsm8k.jsonl"


def test_run_path_adds_the_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "RESULTS_DIR", tmp_path)
    (tmp_path / "demo.jsonl").write_text("{}\n", encoding="utf-8")
    assert mcp_server._run_path("demo") == mcp_server._run_path("demo.jsonl")


def test_missing_run_names_what_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "RESULTS_DIR", tmp_path)
    (tmp_path / "real.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="real"):
        mcp_server._run_path("nope")
