#!/usr/bin/env python3
"""AQUA Dashboard - Generates a self-contained HTML dashboard.

    python3 ~/water-tracker/aqua_dashboard.py        # open in browser
    python3 ~/water-tracker/aqua_dashboard.py --out   # print path only

No server needed. Queries ~/.aqua/water.db, bakes the data into a
single HTML file at ~/.aqua/dashboard.html, and opens it.
"""

import json
import sqlite3
import sys
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path.home() / ".aqua" / "water.db"
OUT_PATH = Path.home() / ".aqua" / "dashboard.html"
TEMPLATE = Path(__file__).parent / "aqua_dashboard.html"

# ---------------------------------------------------------------------------
# Water calculation (mirrors aqua_cli.py parameters)
# ---------------------------------------------------------------------------

def calc_water(inp, out, think=0, model="opus", scenario="central"):
    ENERGY = {
        "haiku":  {"low": 0.15, "central": 0.4,  "high": 1.0},
        "sonnet": {"low": 0.8,  "central": 2.0,  "high": 5.0},
        "opus":   {"low": 1.5,  "central": 4.0,  "high": 10.0},
    }
    WUE   = {"low": 0.5,  "central": 1.08, "high": 1.8}
    GR_C  = {"low": 0.8,  "central": 1.8,  "high": 2.7}
    GR_W  = {"low": 2.0,  "central": 4.5,  "high": 7.0}
    RENEW = {"low": 0.64, "central": 0.50, "high": 0.30}

    e = ENERGY.get(model, ENERGY["opus"])[scenario]
    equiv = (inp or 0) * 0.25 + (out or 0) + (think or 0)
    energy_wh = equiv * e / 3600

    dc = energy_wh * WUE[scenario]
    ic = energy_wh * GR_C[scenario] * (1 - RENEW[scenario])
    iw = energy_wh * GR_W[scenario] * (1 - RENEW[scenario])
    return {"consumption_ml": dc + ic, "withdrawal_ml": dc + iw, "energy_wh": energy_wh}


COMPARISONS = [
    (0.05, "a single raindrop"), (0.26, "a few drops"),
    (1.0, "1/5 of a teaspoon"), (5.0, "a teaspoon"),
    (15.0, "a tablespoon"), (30.0, "a medicine cup"),
    (60.0, "a shot glass"), (120.0, "a quarter cup"),
    (240.0, "a cup of water"), (355.0, "a can of soda"),
    (500.0, "a water bottle"), (750.0, "a wine bottle"),
    (1000.0, "a liter bottle"), (3785.0, "a gallon jug"),
]

def nearest(ml):
    if ml <= 0:
        return "nothing yet"
    for i, (threshold, name) in enumerate(COMPARISONS):
        if ml < threshold:
            return f"about {COMPARISONS[i-1][1]}" if i > 0 else f"less than {name}"
    return f"more than {COMPARISONS[-1][1]}"


# ---------------------------------------------------------------------------
# Benchmarks - fun real-world comparisons
# ---------------------------------------------------------------------------

def compute_benchmarks(ml):
    """Compute fun real-world water comparison metrics."""
    if ml <= 0:
        return {"toilets": 0, "shower_sec": 0, "glasses": 0, "raindrops": 0,
                "bottles": 0, "daily_pct": 0, "faucet_sec": 0, "ice_cubes": 0}
    return {
        "toilets":    round(ml / 6000, 4),        # 6L per flush (US standard)
        "shower_sec": round(ml / (9500/60), 1),    # 9.5 L/min standard head
        "glasses":    round(ml / 240, 2),           # 8oz / 240mL glass
        "raindrops":  int(ml / 0.05),               # ~0.05 mL per raindrop
        "bottles":    round(ml / 500, 2),           # 500mL water bottle
        "daily_pct":  round((ml / 1596610) * 100, 4),  # avg American daily
        "faucet_sec": round(ml / (8300/60), 1),    # kitchen faucet 8.3 L/min
        "ice_cubes":  round(ml / 18, 1),           # ~18mL per ice cube
    }


# ---------------------------------------------------------------------------
# Database queries — parameterized by time period
# ---------------------------------------------------------------------------

def _query_period(conn, start_iso=None, end_iso=None):
    """Query all dashboard data for a specific time range."""
    if start_iso is None:
        time_clause = ""
        time_params = []
    elif end_iso:
        time_clause = "AND s.started_at >= ? AND s.started_at < ?"
        time_params = [start_iso, end_iso]
    else:
        time_clause = "AND s.started_at >= ?"
        time_params = [start_iso]

    # --- sessions ---
    sessions = conn.execute(
        "SELECT s.session_id, s.started_at, s.ended_at, s.model, s.working_dir,"
        " COUNT(e.id) as event_count,"
        " COALESCE(SUM(e.est_input_tokens),0) as ti,"
        " COALESCE(SUM(e.est_output_tokens),0) as to_,"
        " COALESCE(SUM(e.est_thinking_tokens),0) as tt,"
        " COALESCE(SUM(e.agent_sub_turns),0) as ta,"
        " COUNT(CASE WHEN e.tool_name='Task' THEN 1 END) as task_count"
        " FROM sessions s LEFT JOIN events e ON s.session_id=e.session_id"
        f" WHERE 1=1 {time_clause} GROUP BY s.session_id ORDER BY s.started_at DESC",
        time_params).fetchall()

    session_list = []
    total_c = total_w = total_e = 0
    for s in sessions:
        mdl = s["model"] or "opus"
        w = calc_water(s["ti"], s["to_"], s["tt"], mdl)
        at = s["ta"] or 0
        if at > 0:
            aw = calc_water(5000*at, 1500*at, 0, mdl)
            w["consumption_ml"] += aw["consumption_ml"]
            w["withdrawal_ml"] += aw["withdrawal_ml"]
            w["energy_wh"] += aw["energy_wh"]
        total_c += w["consumption_ml"]
        total_w += w["withdrawal_ml"]
        total_e += w["energy_wh"]
        wd = s["working_dir"] or ""
        session_list.append({
            "dir": Path(wd).name if wd else "?",
            "active": s["ended_at"] is None,
            "events": s["event_count"],
            "tasks": s["task_count"],
            "consumption_ml": round(w["consumption_ml"], 3),
            "model": mdl,
            "started": (s["started_at"] or "")[5:16],
        })

    # --- timeline (capped at 1000 points) ---
    evts = conn.execute(
        "SELECT e.timestamp, e.tool_name, e.est_input_tokens as i,"
        " e.est_output_tokens as o, e.est_thinking_tokens as t,"
        " e.agent_sub_turns as a, s.model"
        " FROM events e JOIN sessions s ON e.session_id=s.session_id"
        f" WHERE 1=1 {time_clause} ORDER BY e.timestamp",
        time_params).fetchall()

    timeline = []
    cum = 0.0
    for ev in evts:
        mdl = ev["model"] or "opus"
        w = calc_water(ev["i"], ev["o"], ev["t"], mdl)
        at = ev["a"] or 0
        if at > 0:
            w["consumption_ml"] += calc_water(5000*at, 1500*at, 0, mdl)["consumption_ml"]
        cum += w["consumption_ml"]
        timeline.append({"t": (ev["timestamp"] or "")[5:19],
                         "tool": ev["tool_name"] or "prompt",
                         "ml": round(cum, 3)})
    if len(timeline) > 1000:
        step = max(len(timeline) // 1000, 1)
        last = timeline[-1]
        timeline = timeline[::step]
        if timeline[-1]["t"] != last["t"]:
            timeline.append(last)

    # --- tools ---
    tools_raw = conn.execute(
        "SELECT e.tool_name, COUNT(*) as cnt,"
        " COALESCE(SUM(e.est_input_tokens),0) as i,"
        " COALESCE(SUM(e.est_output_tokens),0) as o,"
        " COALESCE(SUM(e.est_thinking_tokens),0) as t"
        " FROM events e JOIN sessions s ON e.session_id=s.session_id"
        f" WHERE 1=1 {time_clause} AND e.tool_name IS NOT NULL"
        " GROUP BY e.tool_name ORDER BY cnt DESC",
        time_params).fetchall()
    tool_list = []
    for t in tools_raw:
        w = calc_water(t["i"], t["o"], t["t"])
        tool_list.append({"name": t["tool_name"], "count": t["cnt"],
                          "ml": round(w["consumption_ml"], 3)})

    prompt_row = conn.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(e.est_input_tokens),0) as i,"
        " COALESCE(SUM(e.est_output_tokens),0) as o"
        " FROM events e JOIN sessions s ON e.session_id=s.session_id"
        f" WHERE 1=1 {time_clause} AND e.tool_name IS NULL",
        time_params).fetchone()
    if prompt_row and prompt_row["cnt"] > 0:
        pw = calc_water(prompt_row["i"], prompt_row["o"])
        tool_list.append({"name": "prompt", "count": prompt_row["cnt"],
                          "ml": round(pw["consumption_ml"], 3)})
    tool_list.sort(key=lambda x: x["count"], reverse=True)

    # --- 3-scenario uncertainty ---
    ti = sum(s["ti"] for s in sessions)
    to = sum(s["to_"] for s in sessions)
    tt = sum(s["tt"] for s in sessions)
    ta = sum(s["ta"] for s in sessions)
    scenarios = {}
    for sc in ["low", "central", "high"]:
        w = calc_water(ti, to, tt, "opus", sc)
        if ta > 0:
            w["consumption_ml"] += calc_water(5000*ta, 1500*ta, 0, "opus", sc)["consumption_ml"]
        scenarios[sc] = round(w["consumption_ml"], 3)

    return {
        "totals": {
            "consumption_ml": round(total_c, 3),
            "withdrawal_ml": round(total_w, 3),
            "energy_wh": round(total_e, 3),
            "sessions": len(session_list),
            "events": sum(s["events"] for s in session_list),
            "nearest": nearest(total_c),
        },
        "benchmarks": compute_benchmarks(total_c),
        "sessions": session_list,
        "timeline": timeline,
        "tools": tool_list,
        "scenarios": scenarios,
        "raw_tokens": {"input": ti, "output": to, "thinking": tt, "agent_turns": ta},
    }


def _empty_period():
    return {
        "totals": {"consumption_ml": 0, "withdrawal_ml": 0, "energy_wh": 0,
                    "sessions": 0, "events": 0, "nearest": "nothing yet"},
        "benchmarks": compute_benchmarks(0),
        "sessions": [], "timeline": [], "tools": [],
        "scenarios": {"low": 0, "central": 0, "high": 0},
        "raw_tokens": {"input": 0, "output": 0, "thinking": 0, "agent_turns": 0},
    }


def get_dashboard_data():
    if not DB_PATH.exists():
        empty = _empty_period()
        return {
            "periods": {p: empty for p in
                        ["today", "yesterday", "week", "month", "year", "all"]},
            "history": [],
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)
    year_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    periods = {
        "today":     _query_period(conn, today_start.isoformat()),
        "yesterday": _query_period(conn, yesterday_start.isoformat(),
                                   today_start.isoformat()),
        "week":      _query_period(conn, week_start.isoformat()),
        "month":     _query_period(conn, month_start.isoformat()),
        "year":      _query_period(conn, year_start.isoformat()),
        "all":       _query_period(conn),
    }

    # --- history (7 days) ---
    hist = conn.execute(
        "SELECT DATE(s.started_at) as day, COUNT(e.id) as cnt,"
        " COALESCE(SUM(e.est_input_tokens),0) as i,"
        " COALESCE(SUM(e.est_output_tokens),0) as o,"
        " COALESCE(SUM(e.est_thinking_tokens),0) as t,"
        " COALESCE(SUM(e.agent_sub_turns),0) as a"
        " FROM sessions s LEFT JOIN events e ON s.session_id=e.session_id"
        " WHERE s.started_at>? GROUP BY DATE(s.started_at) ORDER BY day",
        (week_start.isoformat(),)).fetchall()
    history = []
    for h in hist:
        w = calc_water(h["i"], h["o"], h["t"])
        at = h["a"] or 0
        if at > 0:
            w["consumption_ml"] += calc_water(5000*at, 1500*at, 0)["consumption_ml"]
        history.append({"date": h["day"], "ml": round(w["consumption_ml"], 2),
                        "events": h["cnt"]})

    conn.close()

    return {
        "periods": periods,
        "history": history,
        "generated": now.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Generate static HTML
# ---------------------------------------------------------------------------

def generate():
    data = get_dashboard_data()
    template = TEMPLATE.read_text()
    html = template.replace('"__DATA_PLACEHOLDER__"', json.dumps(data))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)
    return OUT_PATH


def main():
    out = generate()
    if "--out" in sys.argv:
        print(str(out))
    else:
        print(f"\n  AQUA Dashboard -> {out}")
        webbrowser.open(f"file://{out}")
        print(f"  Re-run to refresh data.\n")


if __name__ == "__main__":
    main()
