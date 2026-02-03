#!/usr/bin/env python3
"""
AQUA Hook - Lightweight water tracking for Claude Code hooks.

Logs each interaction to ~/.aqua/water.db (SQLite, WAL mode).
Designed to be fast (<50ms) and crash-safe.

Usage in Claude Code hooks:
  PostToolUse:    python3 ~/water-tracker/aqua_hook.py post-tool
  UserPromptSubmit: python3 ~/water-tracker/aqua_hook.py prompt
  Stop:           python3 ~/water-tracker/aqua_hook.py stop
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

AQUA_DIR = Path.home() / ".aqua"
DB_PATH = AQUA_DIR / "water.db"

# Token estimates by tool type (marginal cost per tool call in a response)
# These represent the approximate inference work per tool-use cycle.
TOOL_TOKEN_ESTIMATES = {
    # Tool name:    (est_input, est_output, est_thinking)
    "Read":         (800,   100,  0),
    "Glob":         (600,   150,  0),
    "Grep":         (600,   150,  0),
    "Edit":         (2000,  600,  0),
    "Write":        (2000,  500,  0),
    "Bash":         (1500,  500,  0),
    "Task":         (3000,  1500, 0),    # Just the spawn; sub-agent tracked separately
    "WebFetch":     (1500,  400,  0),
    "WebSearch":    (1000,  400,  0),
    "AskUserQuestion": (1000, 300, 0),
    "Skill":        (1000,  300,  0),
    "NotebookEdit": (1500,  500,  0),
}
DEFAULT_TOOL_ESTIMATE = (1000, 300, 0)

# Sub-agent token estimates (per turn within a Task agent)
AGENT_TOKENS_PER_TURN = (5000, 1500, 0)


def init_db():
    """Initialize SQLite database with WAL mode for concurrent access."""
    AQUA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            pid INTEGER,
            ppid INTEGER,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            model TEXT DEFAULT 'opus',
            working_dir TEXT,
            instance_label TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            tool_name TEXT,
            est_input_tokens INTEGER DEFAULT 0,
            est_output_tokens INTEGER DEFAULT 0,
            est_thinking_tokens INTEGER DEFAULT 0,
            agent_sub_turns INTEGER DEFAULT 0,
            metadata TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
    """)
    return conn


def get_or_create_session(conn):
    """Get active session for current Claude Code instance, or create one."""
    ppid = os.getppid()
    pid = os.getpid()

    # Check for active session with this PPID (Claude Code process)
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE ppid = ? AND ended_at IS NULL "
        "ORDER BY started_at DESC LIMIT 1",
        (ppid,)
    ).fetchone()

    if row:
        return row[0]

    # Create new session
    now = datetime.now()
    session_id = f"{ppid}_{now.strftime('%Y%m%d_%H%M%S')}"
    cwd = os.environ.get("PWD", os.getcwd())
    model = os.environ.get("AQUA_MODEL", "opus")

    conn.execute(
        "INSERT INTO sessions (session_id, pid, ppid, started_at, model, working_dir) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, pid, ppid, now.isoformat(), model, cwd)
    )
    conn.commit()
    return session_id


def log_event(conn, session_id, event_type, tool_name=None,
              est_in=0, est_out=0, est_think=0, agent_turns=0, metadata=None):
    """Log a tracking event."""
    conn.execute(
        "INSERT INTO events (session_id, timestamp, event_type, tool_name, "
        "est_input_tokens, est_output_tokens, est_thinking_tokens, "
        "agent_sub_turns, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, datetime.now().isoformat(), event_type, tool_name,
         est_in, est_out, est_think, agent_turns,
         json.dumps(metadata) if metadata else None)
    )
    conn.commit()


def handle_post_tool(input_data):
    """Handle PostToolUse event."""
    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})

    est_in, est_out, est_think = TOOL_TOKEN_ESTIMATES.get(
        tool_name, DEFAULT_TOOL_ESTIMATE
    )

    # Estimate sub-agent turns for Task tool
    agent_turns = 0
    if tool_name == "Task":
        max_turns = tool_input.get("max_turns", 10)
        agent_turns = min(max_turns, 15)  # Cap estimate at 15

    conn = init_db()
    try:
        session_id = get_or_create_session(conn)
        meta = {"tool_input_keys": list(tool_input.keys())[:5]}
        if tool_name == "Task":
            meta["subagent_type"] = tool_input.get("subagent_type", "unknown")
            meta["agent_model"] = tool_input.get("model", "inherited")

        log_event(conn, session_id, "tool_use", tool_name,
                  est_in, est_out, est_think, agent_turns, meta)
    finally:
        conn.close()


def handle_prompt(input_data):
    """Handle UserPromptSubmit event."""
    conn = init_db()
    try:
        session_id = get_or_create_session(conn)
        # Each user prompt = start of a new assistant response cycle
        # Estimate: system prompt re-processing + user message
        log_event(conn, session_id, "prompt", None,
                  est_in=2000, est_out=500, est_think=0)
    finally:
        conn.close()


def handle_stop(input_data):
    """Handle Stop event - finalize session."""
    conn = init_db()
    try:
        ppid = os.getppid()
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE ppid = ? AND ended_at IS NULL",
            (datetime.now().isoformat(), ppid)
        )
        conn.commit()
    finally:
        conn.close()


DASH_TS = AQUA_DIR / ".dash_ts"
DASH_SCRIPT = Path(__file__).parent / "aqua_dashboard.py"


def maybe_regen_dashboard():
    """Regenerate dashboard HTML in background, throttled to every 15s."""
    try:
        if not DASH_SCRIPT.exists():
            return
        if DASH_TS.exists():
            age = time.time() - DASH_TS.stat().st_mtime
            if age < 15:
                return
        DASH_TS.touch()
        import subprocess
        subprocess.Popen(
            [sys.executable, str(DASH_SCRIPT), "--out"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def main():
    event_type = sys.argv[1] if len(sys.argv) > 1 else "post-tool"

    # Read stdin
    input_data = {}
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                input_data = json.loads(raw)
    except Exception:
        pass

    try:
        if event_type == "post-tool":
            handle_post_tool(input_data)
        elif event_type == "prompt":
            handle_prompt(input_data)
        elif event_type == "stop":
            handle_stop(input_data)
    except Exception:
        pass  # Never crash Claude Code

    # Regenerate dashboard in background (throttled to every 15s)
    maybe_regen_dashboard()

    # Output empty JSON (no messages to inject)
    print(json.dumps({}))


if __name__ == "__main__":
    main()
