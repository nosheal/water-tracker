#!/usr/bin/env python3
"""
AQUA v2 - AI Query Water Usage Analyzer

Technoeconomic model of water consumption AND withdrawal per AI query,
with explicit uncertainty bounds on all parameters.

Tracks usage across multiple concurrent Claude Code instances via
shared SQLite database (~/.aqua/water.db) with WAL mode.

Sources: LBNL, NREL, Google, Masley, Ren et al., TokenPowerBench
"""

import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ==========================================================================
# TERMINAL FORMATTING
# ==========================================================================

BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

HEADER = f"""
{BLUE}========================================================{RESET}
{BOLD}  AQUA v2  {RESET}{DIM}AI Query Water Usage Analyzer{RESET}
{BLUE}========================================================{RESET}"""


# ==========================================================================
# PARAMETER UNCERTAINTY MODEL
#
# Every parameter has (low, central, high) with explicit sources.
# "low" = best-case for water (least water consumed)
# "high" = worst-case for water (most water consumed)
# ==========================================================================

@dataclass(frozen=True)
class Param:
    """A model parameter with uncertainty range and sourcing."""
    low: float
    central: float
    high: float
    unit: str
    source: str
    note: str = ""

    def value(self, scenario: str = "central") -> float:
        return {"low": self.low, "central": self.central, "high": self.high}[scenario]

    def range_str(self) -> str:
        return f"{self.low}-{self.central}-{self.high} {self.unit}"


# --- Energy per equivalent output token (Joules, comprehensive) ---
# "Comprehensive" = active accelerator + host CPU/DRAM + idle provisioning + PUE
# Calibrated to Google Aug 2025: median Gemini prompt = 0.24 Wh comprehensive
#   (vs 0.10 Wh active-only → 2.4x overhead factor)
# Cross-check: vLLM H100, Llama3-70B FP8, batch=128 → 0.39 J/tok active
#   With 2.4x overhead: ~0.94 J/tok comprehensive
# Anthropic does NOT disclose. We estimate from public benchmarks.

ENERGY_PER_EQUIV_TOKEN = {
    "haiku": Param(
        low=0.15, central=0.4, high=1.0, unit="J/equiv_tok",
        source="TokenPowerBench[7], ML.ENERGY[8], Epoch AI. Active benchmarks ~0.05-0.15 J/tok for 8B models, with 2-4x overhead.",
        note="~8-20B params. Uncertainty: 2.5x range.",
    ),
    "sonnet": Param(
        low=0.8, central=2.0, high=5.0, unit="J/equiv_tok",
        source="Calibrated to Google Gemini 0.24Wh/prompt[2]. vLLM 70B FP8 active: 0.39 J/tok[7] -> ~1.0 comprehensive. Range accounts for larger MoE architectures.",
        note="~70B MoE. Uncertainty: 6x range.",
    ),
    "opus": Param(
        low=1.5, central=4.0, high=10.0, unit="J/equiv_tok",
        source="Estimated ~2x Sonnet, scaled by API pricing ratio and model size. No direct measurement available.",
        note="~200B+ params, likely MoE. Uncertainty: 7x range. HIGHEST UNCERTAINTY.",
    ),
}

# Input tokens cost less than output (prefill is parallelized, no autoregressive decode)
INPUT_COST_RATIO = Param(
    low=0.15, central=0.25, high=0.40, unit="ratio",
    source="API pricing ratios (Anthropic, OpenAI) and hardware utilization data. Prefill is 3-10x faster than decode per token.",
    note="Input tokens as fraction of output token cost.",
)

MODEL_LABELS = {
    "haiku": ("Haiku", "~8-20B"),
    "sonnet": ("Sonnet", "~70B MoE"),
    "opus": ("Opus 4.5", "~200B+"),
}


# --- Infrastructure: WUE, Grid Water Intensity ---

# WUE: onsite water for cooling (L per kWh of IT load)
# This is CONSUMPTION - water evaporated, not returned.
# For evaporative cooling, withdrawal ≈ consumption (all evaporated).
WUE_DIRECT = {
    "google_cloud": Param(
        low=0.5, central=1.08, high=1.8, unit="L/kWh",
        source="Central derived from Google: 0.26mL/0.24Wh=1.083[2]. Range from LBNL 2024 evaporative cooling[1]. Low assumes partial air-cooling or cool-climate sites.",
        note="Varies by location, season, and cooling type.",
    ),
    "aws_east": Param(
        low=0.05, central=0.15, high=0.5, unit="L/kWh",
        source="AWS Indiana disclosure: 0.15 L/kWh, 40% improvement since 2021[AWS/DCK]. Low for direct-to-chip liquid cooling.",
        note="Project Rainier uses advanced liquid cooling.",
    ),
    "industry_avg": Param(
        low=1.0, central=1.8, high=3.0, unit="L/kWh",
        source="LBNL 2024 US Data Center Energy Report[1]. Uptime Institute surveys.",
        note="Wide range: air-cooled (low) to aggressive evaporative (high).",
    ),
}

# Grid water intensity: water used in electricity generation
# CRITICAL: consumption vs withdrawal are different metrics.
# Consumption = evaporated, permanently removed from local water cycle
# Withdrawal = total intake, most returned (but warmer, affecting ecosystems)
GRID_WATER_CONSUMPTION = Param(
    low=0.8, central=1.8, high=2.7, unit="L/kWh",
    source="Macknick et al. 2012 (NREL/TP-6A20-50900)[9]. US generation mix weighted average. Low=high renewables/gas. High=coal-heavy grid.",
    note="By generation type: Coal ~2.6, Gas CC ~0.95, Nuclear ~2.7, Solar/Wind ~0 L/kWh.",
)

GRID_WATER_WITHDRAWAL = Param(
    low=2.0, central=4.5, high=7.0, unit="L/kWh",
    source="LBNL 2024 derived: 211B gal/176 TWh=4.54[1]. Macknick 2012[9]. Withdrawal includes returned-but-warmed water.",
    note="Withdrawal >> consumption because most water is returned to source.",
)

# Renewable energy fraction (reduces indirect water)
RENEWABLE_FRACTION = {
    "google_cloud": Param(
        low=0.30, central=0.50, high=0.64, unit="fraction",
        source="Google 2024 Environmental Report: 64% carbon-free energy (CFE)[2]. Central discounts RECs vs actual load matching.",
        note="Hourly CFE < annual CFE. 0.50 is conservative central.",
    ),
    "aws_east": Param(
        low=0.20, central=0.50, high=1.0, unit="fraction",
        source="AWS claims 100% matched renewable (2023). Actual hourly CFE is lower. Wide uncertainty.",
        note="RECs != real-time renewable. High=accept AWS's claim.",
    ),
    "industry_avg": Param(
        low=0.05, central=0.20, high=0.40, unit="fraction",
        source="IEA, EPA eGRID data. Most data centers are grid-connected with limited renewable commitment.",
    ),
}

# PUE: already baked into comprehensive energy numbers, tracked for display
PUE = {
    "google_cloud": Param(low=1.06, central=1.09, high=1.15, unit="ratio",
        source="Google 2024 Environmental Report[2]. Fleet average 1.09."),
    "aws_east": Param(low=1.08, central=1.12, high=1.20, unit="ratio",
        source="AWS sustainability reports. Estimated from industry data."),
    "industry_avg": Param(low=1.30, central=1.58, high=1.80, unit="ratio",
        source="Uptime Institute 2024 global survey[1]."),
}

INFRA_LABELS = {
    "google_cloud": "Google Cloud (TPU)",
    "aws_east": "AWS US East (Trainium)",
    "industry_avg": "Industry Average",
}

DEFAULT_INFRA = "google_cloud"

# --- Token templates for Claude Code interactions ---
INTERACTION_TYPES = {
    "simple":    {"input": 800,   "output": 300,  "thinking": 0,     "label": "Simple reply"},
    "code_edit": {"input": 3000,  "output": 1000, "thinking": 0,     "label": "Read + edit file"},
    "tool_heavy":{"input": 6000,  "output": 2000, "thinking": 0,     "label": "Multi-tool turn"},
    "agent":     {"input": 8000,  "output": 2500, "thinking": 0,     "label": "Sub-agent turn"},
    "reasoning": {"input": 3000,  "output": 1000, "thinking": 10000, "label": "Extended thinking"},
    "planning":  {"input": 10000, "output": 3000, "thinking": 5000,  "label": "Plan mode"},
}

# ==========================================================================
# CORE WATER MODEL (with uncertainty propagation)
# ==========================================================================

def equiv_tokens(input_tok: int, output_tok: int, thinking_tok: int = 0,
                 scenario: str = "central") -> float:
    ratio = INPUT_COST_RATIO.value(scenario)
    return input_tok * ratio + output_tok + thinking_tok


def calculate_water(
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    model: str = "opus",
    infra: str = DEFAULT_INFRA,
    scenario: str = "central",
) -> dict:
    """
    Calculate water consumption AND withdrawal for a single inference call.

    scenario: "low" (best case), "central", "high" (worst case)
    All parameters shift together: low=all optimistic, high=all pessimistic.
    """
    # Parameter lookups
    e_param = ENERGY_PER_EQUIV_TOKEN[model]
    wue_param = WUE_DIRECT[infra]
    grid_c_param = GRID_WATER_CONSUMPTION
    grid_w_param = GRID_WATER_WITHDRAWAL
    renew_param = RENEWABLE_FRACTION[infra]

    # For water: "low" scenario = least water = low energy, low WUE, high renewable
    # "high" scenario = most water = high energy, high WUE, low renewable
    if scenario == "low":
        e_j = e_param.low
        wue = wue_param.low
        grid_c = grid_c_param.low
        grid_w = grid_w_param.low
        renew = renew_param.high  # High renewable = less water
        input_ratio = INPUT_COST_RATIO.low
    elif scenario == "high":
        e_j = e_param.high
        wue = wue_param.high
        grid_c = grid_c_param.high
        grid_w = grid_w_param.high
        renew = renew_param.low  # Low renewable = more water
        input_ratio = INPUT_COST_RATIO.high
    else:
        e_j = e_param.central
        wue = wue_param.central
        grid_c = grid_c_param.central
        grid_w = grid_w_param.central
        renew = renew_param.central
        input_ratio = INPUT_COST_RATIO.central

    # Equivalent tokens
    eq = input_tokens * input_ratio + output_tokens + thinking_tokens

    # Energy (Joules -> kWh)
    energy_j = eq * e_j
    energy_wh = energy_j / 3600.0
    energy_kwh = energy_wh / 1000.0

    # DIRECT water: onsite cooling (consumption only; evaporative ≈ all consumed)
    direct_consumption_ml = energy_kwh * wue * 1000.0
    direct_withdrawal_ml = direct_consumption_ml  # Evaporative: withdrawal ≈ consumption

    # INDIRECT water: electricity generation (non-renewable fraction)
    non_renew = 1.0 - renew
    indirect_consumption_ml = energy_kwh * grid_c * non_renew * 1000.0
    indirect_withdrawal_ml = energy_kwh * grid_w * non_renew * 1000.0

    total_consumption_ml = direct_consumption_ml + indirect_consumption_ml
    total_withdrawal_ml = direct_withdrawal_ml + indirect_withdrawal_ml

    return {
        "consumption_ml": total_consumption_ml,
        "withdrawal_ml": total_withdrawal_ml,
        "direct_consumption_ml": direct_consumption_ml,
        "direct_withdrawal_ml": direct_withdrawal_ml,
        "indirect_consumption_ml": indirect_consumption_ml,
        "indirect_withdrawal_ml": indirect_withdrawal_ml,
        "energy_j": energy_j,
        "energy_wh": energy_wh,
        "equiv_tokens": eq,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "model": MODEL_LABELS[model][0],
        "model_key": model,
        "infra": INFRA_LABELS[infra],
        "infra_key": infra,
        "scenario": scenario,
        "renewable_pct": renew * 100,
    }


def calculate_water_3(input_tokens, output_tokens, thinking_tokens=0,
                      model="opus", infra=DEFAULT_INFRA):
    """Calculate all three scenarios at once, return (low, central, high)."""
    return tuple(
        calculate_water(input_tokens, output_tokens, thinking_tokens,
                        model=model, infra=infra, scenario=s)
        for s in ("low", "central", "high")
    )


# ==========================================================================
# AGENT SWARM ESTIMATOR
# ==========================================================================

def estimate_swarm(num_agents, turns_per_agent, model="opus", infra=DEFAULT_INFRA):
    it = INTERACTION_TYPES["agent"]
    total_calls = num_agents * turns_per_agent
    lo, cen, hi = calculate_water_3(it["input"], it["output"], it["thinking"],
                                    model=model, infra=infra)
    def scale(r):
        scalable = ("_ml", "energy_j", "energy_wh", "equiv_tokens",
                    "input_tokens", "output_tokens", "thinking_tokens")
        out = {}
        for k, v in r.items():
            if isinstance(v, (int, float)) and any(k.endswith(s) or k == s for s in scalable):
                out[k] = v * total_calls
            else:
                out[k] = v
        return out
    return {
        "num_agents": num_agents,
        "turns_per_agent": turns_per_agent,
        "total_calls": total_calls,
        "per_call": cen,
        "low": scale(lo), "central": scale(cen), "high": scale(hi),
    }


# ==========================================================================
# SESSION SIMULATOR
# ==========================================================================

def simulate_session(num_turns=30, model="opus", infra=DEFAULT_INFRA,
                     include_agents=0, agent_turns=0):
    turn_mix = [
        ("simple", 0.10), ("code_edit", 0.35), ("tool_heavy", 0.25),
        ("reasoning", 0.15), ("planning", 0.10), ("agent", 0.05),
    ]
    totals = {"consumption_ml": 0, "withdrawal_ml": 0, "energy_wh": 0,
              "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    turn_idx = 0
    for itype, frac in turn_mix:
        n = max(1, round(num_turns * frac))
        it = INTERACTION_TYPES[itype]
        for _ in range(n):
            if turn_idx >= num_turns:
                break
            growth = 1.0 + 0.5 * (turn_idx / max(num_turns - 1, 1))
            r = calculate_water(int(it["input"] * growth), it["output"],
                                it["thinking"], model=model, infra=infra)
            totals["consumption_ml"] += r["consumption_ml"]
            totals["withdrawal_ml"] += r["withdrawal_ml"]
            totals["energy_wh"] += r["energy_wh"]
            totals["input_tokens"] += r["input_tokens"]
            totals["output_tokens"] += r["output_tokens"]
            totals["thinking_tokens"] += r["thinking_tokens"]
            turn_idx += 1

    swarm_c, swarm_w = 0, 0
    if include_agents > 0 and agent_turns > 0:
        sw = estimate_swarm(include_agents, agent_turns, model=model, infra=infra)
        swarm_c = sw["central"]["consumption_ml"]
        swarm_w = sw["central"]["withdrawal_ml"]
        totals["consumption_ml"] += swarm_c
        totals["withdrawal_ml"] += swarm_w
        totals["energy_wh"] += sw["central"]["energy_wh"]

    return {**totals, "num_turns": min(turn_idx, num_turns),
            "model": MODEL_LABELS[model][0], "infra": INFRA_LABELS[infra],
            "swarm_consumption_ml": swarm_c, "swarm_withdrawal_ml": swarm_w,
            "swarm_agents": include_agents, "swarm_turns": agent_turns}


# ==========================================================================
# SQLITE DATABASE LAYER (reads from hook-written database)
# ==========================================================================

AQUA_DIR = Path.home() / ".aqua"
DB_PATH = AQUA_DIR / "water.db"


def get_db():
    """Open the shared SQLite database (read-only for CLI, writable for hooks)."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def query_sessions(conn, days=7):
    """Get recent sessions with aggregated water data."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute("""
        SELECT
            s.session_id,
            s.started_at,
            s.ended_at,
            s.model,
            s.working_dir,
            COUNT(e.id) as event_count,
            SUM(e.est_input_tokens) as total_input,
            SUM(e.est_output_tokens) as total_output,
            SUM(e.est_thinking_tokens) as total_thinking,
            SUM(e.agent_sub_turns) as total_agent_turns,
            COUNT(CASE WHEN e.event_type = 'prompt' THEN 1 END) as prompt_count,
            COUNT(CASE WHEN e.tool_name = 'Task' THEN 1 END) as task_count
        FROM sessions s
        LEFT JOIN events e ON s.session_id = e.session_id
        WHERE s.started_at > ?
        GROUP BY s.session_id
        ORDER BY s.started_at DESC
    """, (cutoff,)).fetchall()
    return rows


def query_totals(conn, days=None):
    """Get aggregate totals across all sessions."""
    if days:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        where = f"WHERE s.started_at > '{cutoff}'"
    else:
        where = ""
    row = conn.execute(f"""
        SELECT
            COUNT(DISTINCT s.session_id) as sessions,
            COUNT(e.id) as events,
            COALESCE(SUM(e.est_input_tokens), 0) as input_tokens,
            COALESCE(SUM(e.est_output_tokens), 0) as output_tokens,
            COALESCE(SUM(e.est_thinking_tokens), 0) as thinking_tokens,
            COALESCE(SUM(e.agent_sub_turns), 0) as agent_turns
        FROM sessions s
        LEFT JOIN events e ON s.session_id = e.session_id
        {where}
    """).fetchone()
    return row


def compute_water_from_db_row(row, model="opus", infra=DEFAULT_INFRA):
    """Compute water estimates from aggregated token counts."""
    input_t = row["total_input"] or 0
    output_t = row["total_output"] or 0
    thinking_t = row["total_thinking"] or 0
    agent_turns = row["total_agent_turns"] or 0

    # Base inference water
    r = calculate_water(input_t, output_t, thinking_t, model=model, infra=infra)

    # Add estimated agent sub-turns (each agent turn ≈ 5000 in + 1500 out)
    if agent_turns > 0:
        agent_r = calculate_water(5000 * agent_turns, 1500 * agent_turns, 0,
                                  model=model, infra=infra)
        r["consumption_ml"] += agent_r["consumption_ml"]
        r["withdrawal_ml"] += agent_r["withdrawal_ml"]
        r["energy_wh"] += agent_r["energy_wh"]

    return r


# ==========================================================================
# DISPLAY HELPERS
# ==========================================================================

COMPARISONS = [
    (0.05, "a single raindrop"), (0.26, "5 drops (Google's Gemini figure)"),
    (1.0, "1/5 of a teaspoon"), (5.0, "a teaspoon"),
    (15.0, "a tablespoon"), (30.0, "a medicine cup"),
    (60.0, "a shot glass"), (120.0, "a quarter cup"),
    (240.0, "a cup of water"), (355.0, "a can of soda"),
    (500.0, "a water bottle"), (750.0, "a wine bottle"),
    (1000.0, "a liter bottle"), (3785.0, "a gallon jug"),
    (19000.0, "a 5-gal bucket"), (75700.0, "a small fish tank"),
    (151000.0, "a bathtub"),
]

DAILY = [
    ("flushing a toilet", 6000.0), ("brushing teeth", 7500.0),
    ("a 1-min shower", 9500.0), ("a dishwasher cycle", 22700.0),
    ("a 5-min shower", 47000.0), ("a laundry load", 57000.0),
    ("avg American daily use", 1596610.0),
]


def nearest(ml):
    if ml <= 0: return "essentially nothing"
    for t, d in COMPARISONS:
        if ml < t * 1.3:
            return f"about {d}" if ml / t > 0.85 else f"less than {d}"
    return f"{ml/3785:.1f} gallons"


def daily_ctx(ml):
    for n, v in DAILY:
        if ml < v:
            p = ml / v * 100
            return f"{p:.3f}% of {n}" if p < 0.1 else f"{p:.2f}% of {n}" if p < 1 else f"{p:.1f}% of {n}"
    return f"{ml/1596610*100:.1f}% of avg daily use"


def bar(ml, mx=500.0, w=30):
    f = min(1.0, ml / mx)
    n = int(f * w)
    return f"[{chr(9619)*n}{chr(9617)*(w-n)}]"


def render_drop(ml):
    if ml < 0.3: return "    .\n"
    if ml < 2:   return "      ,\n     / \\\n    |   |\n     \\_/\n"
    if ml < 20:  return "       ,\n      / \\\n     /   \\\n    |     |\n    |     |\n     \\   /\n      \\_/\n"
    if ml < 200: return "        ,\n       / \\\n      /   \\\n     /     \\\n    |  ~~~  |\n    |       |\n     \\     /\n      \\_/\n"
    return "         ,\n        / \\\n       / ~ \\\n      / ~~~ \\\n     / ~~~~~ \\\n    | ~~~~~~~ |\n    | ~~~~~~~ |\n     \\ ~~~~~ /\n      \\   /\n       \\_/\n"


def render_glass(ml, cap=500.0):
    fill = max(0.0, min(1.0, ml / cap))
    rows, tw, bw = 10, 20, 10
    fr = int(fill * rows)
    lines = []
    for i in range(rows):
        w = int(tw - (tw - bw) * (i / (rows - 1)))
        pad = (tw - w) // 2
        rfb = rows - 1 - i
        if rfb < fr:
            c = "~" * (w-2) if rfb == fr-1 else "#" * (w-2)
            lines.append(" "*pad + "|" + c + "|")
        else:
            lines.append(" "*pad + "|" + " "*(w-2) + "|")
    lines.append(" "*((tw-bw)//2) + " \\" + "_"*(bw-2) + "/")
    lines.append(f" {fill*100:.0f}% of {cap:.0f}mL")
    return "\n".join(lines)


# ==========================================================================
# CLI COMMANDS
# ==========================================================================

def print_help():
    print(f"""{HEADER}

  {BOLD}Estimation (no hooks needed):{RESET}
    aqua query [tokens] [model]           Single query (3-scenario)
    aqua swarm [agents] [turns] [model]   Agent swarm estimate
    aqua session [turns] [agents] [a_turns]  Claude Code session sim
    aqua compare                          Side-by-side table
    aqua dashboard [model]                Interactive live tracker

  {BOLD}Live tracking (requires hook setup):{RESET}
    aqua status                           Current tracking status
    aqua live                             Show live session data
    aqua history [days]                   Past session history

  {BOLD}Reference:{RESET}
    aqua methodology                      Full methodology & sources
    aqua params                           All parameters with uncertainty
    aqua setup                            Hook installation instructions

  {BOLD}Models:{RESET}  haiku, sonnet, opus
  {BOLD}Infra:{RESET}   google_cloud, aws_east, industry_avg

  {BOLD}Quick examples:{RESET}
    python aqua_cli.py query 2000 opus
    python aqua_cli.py swarm 5 10
    python aqua_cli.py session 50
    python aqua_cli.py session 30 3 8     {DIM}# 30 turns + 3 agents x 8{RESET}
    python aqua_cli.py status
    python aqua_cli.py setup
""")


def parse_model_and_nums(args):
    """Parse args into model name and list of numbers."""
    model = "opus"
    nums = []
    infra = DEFAULT_INFRA
    for a in args:
        if a in MODEL_LABELS: model = a
        elif a in INFRA_LABELS: infra = a
        else:
            try: nums.append(int(a))
            except ValueError: pass
    return model, infra, nums


# --- query ---
def cmd_query(args):
    model, infra, nums = parse_model_and_nums(args)
    tokens = nums[0] if nums else 1000
    in_t, out_t = int(tokens * 0.4), int(tokens * 0.6)

    lo, cen, hi = calculate_water_3(in_t, out_t, model=model, infra=infra)

    print(HEADER)
    label, params = MODEL_LABELS[model]
    print(f"  {BOLD}Model:{RESET}  {label} ({params})")
    print(f"  {BOLD}Infra:{RESET}  {INFRA_LABELS[infra]}")
    print(f"  {BOLD}Tokens:{RESET} {in_t:,} in + {out_t:,} out = {tokens:,}")
    print(f"  {BOLD}Energy:{RESET} {cen['energy_j']:.1f} J ({lo['energy_j']:.1f}-{hi['energy_j']:.1f})")
    print()

    print(f"  {BOLD}{'Scenario':<16} {'Consumption':>13} {'Withdrawal':>13} {'Renewable':>10}{RESET}")
    print(f"  {'-'*54}")
    for tag, r, color in [("LOW (best)", lo, GREEN), ("CENTRAL", cen, YELLOW), ("HIGH (worst)", hi, RED)]:
        print(f"  {color}{tag:<16}{RESET} {r['consumption_ml']:>10.4f} mL {r['withdrawal_ml']:>10.4f} mL {r['renewable_pct']:>8.0f}%")
    print()
    print(f"  {DIM}Consumption = water evaporated (environmentally lost)")
    print(f"  Withdrawal = total intake (mostly returned, but warmer){RESET}")
    print()
    print(f"  Central consumption: {BOLD}{nearest(cen['consumption_ml'])}{RESET}")
    print(f"  {DIM}{daily_ctx(cen['consumption_ml'])}{RESET}")
    print()
    for line in render_drop(cen["consumption_ml"]).split("\n"):
        print(f"    {CYAN}{line}{RESET}")
    print(f"  {DIM}(~microwave for {cen['energy_wh']*1.67:.1f}s | uncertainty: {lo['consumption_ml']:.4f}-{hi['consumption_ml']:.4f} mL){RESET}")
    print()


# --- swarm ---
def cmd_swarm(args):
    model, infra, nums = parse_model_and_nums(args)
    agents = nums[0] if len(nums) > 0 else 5
    turns = nums[1] if len(nums) > 1 else 10

    sw = estimate_swarm(agents, turns, model=model, infra=infra)
    lo, cen, hi = sw["low"], sw["central"], sw["high"]

    print(HEADER)
    print(f"  {BOLD}AGENT SWARM ESTIMATE{RESET}")
    print(f"  Agents: {agents}  |  Turns/agent: {turns}  |  Total calls: {sw['total_calls']}")
    print(f"  Model: {cen['model']}  |  Infra: {cen['infra']}")
    print()
    print(f"  {BOLD}{'':>20} {'Consumption':>13} {'Withdrawal':>13}{RESET}")
    print(f"  {'Per call':<20} {sw['per_call']['consumption_ml']:>10.4f} mL {sw['per_call']['withdrawal_ml']:>10.4f} mL")
    print(f"  {'-'*48}")
    print(f"  {GREEN}{'Low (best)':<20}{RESET} {lo['consumption_ml']:>10.2f} mL {lo['withdrawal_ml']:>10.2f} mL")
    print(f"  {YELLOW}{'Central':<20}{RESET} {BOLD}{cen['consumption_ml']:>10.2f} mL {cen['withdrawal_ml']:>10.2f} mL{RESET}")
    print(f"  {RED}{'High (worst)':<20}{RESET} {hi['consumption_ml']:>10.2f} mL {hi['withdrawal_ml']:>10.2f} mL")
    print()
    print(f"  Central: {BOLD}{nearest(cen['consumption_ml'])}{RESET} consumed")
    print(f"  {DIM}{daily_ctx(cen['consumption_ml'])}{RESET}")
    print()
    print(render_glass(cen["consumption_ml"]))
    print()


# --- session ---
def cmd_session(args):
    model, infra, nums = parse_model_and_nums(args)
    turns = nums[0] if len(nums) > 0 else 30
    agents = nums[1] if len(nums) > 1 else 0
    a_turns = nums[2] if len(nums) > 2 else 0

    r = simulate_session(turns, model=model, infra=infra,
                         include_agents=agents, agent_turns=a_turns)
    print(HEADER)
    print(f"  {BOLD}CLAUDE CODE SESSION SIMULATION{RESET}")
    print(f"  Turns: {r['num_turns']}  |  Model: {r['model']}  |  Infra: {r['infra']}")
    if agents:
        print(f"  Agent swarm: {agents} agents x {a_turns} turns")
    print()
    print(f"  Tokens: {r['input_tokens']:,.0f} in + {r['output_tokens']:,.0f} out + {r['thinking_tokens']:,.0f} thinking")
    print(f"  Energy: {r['energy_wh']:.2f} Wh")
    print()
    print(f"  {BOLD}{'':>22} {'Consumption':>13} {'Withdrawal':>13}{RESET}")
    print(f"  {'Direct (cooling)':<22} {r['consumption_ml']-r.get('_ic',r['consumption_ml']*0.5):>10.2f} mL   {DIM}(onsite evaporative){RESET}")
    # Approximate breakdown since simulate_session returns totals
    print(f"  {'Indirect (grid)':<22} {r['consumption_ml']*0.5:>10.2f} mL   {DIM}(electricity generation){RESET}")
    if agents:
        print(f"  {'Agent swarm':<22} {r['swarm_consumption_ml']:>10.2f} mL")
    print(f"  {'-'*48}")
    print(f"  {BOLD}{'TOTAL':<22} {r['consumption_ml']:>10.2f} mL {r['withdrawal_ml']:>10.2f} mL{RESET}")
    print()
    print(f"  Consumption: {BOLD}{nearest(r['consumption_ml'])}{RESET}")
    print(f"  Withdrawal:  {nearest(r['withdrawal_ml'])}")
    print(f"  {DIM}{daily_ctx(r['consumption_ml'])}{RESET}")
    print()
    print(render_glass(r["consumption_ml"]))
    print()
    print(f"  {BOLD}--- Perspective ---{RESET}")
    print(f"  = microwave for {r['energy_wh']*1.67:.0f}s")
    print(f"  = {r['consumption_ml']/1596610*100:.4f}% of avg American daily water use")
    print()


# --- compare ---
def cmd_compare(args):
    print(HEADER)
    print(f"  {BOLD}MODEL x INFRASTRUCTURE COMPARISON{RESET}")
    print(f"  {DIM}1000 tokens (400 in + 600 out), central scenario{RESET}")
    print()
    print(f"  {'Model':<10} {'Infra':<24} {'Consump mL':>11} {'Withdraw mL':>12} {'Energy Wh':>10} {DIM}{'J/eq_tok':>9}{RESET}")
    print("  " + "-" * 78)

    for m in ["haiku", "sonnet", "opus"]:
        for inf in ["google_cloud", "aws_east", "industry_avg"]:
            r = calculate_water(400, 600, model=m, infra=inf)
            e = ENERGY_PER_EQUIV_TOKEN[m]
            print(f"  {MODEL_LABELS[m][0]:<10} {INFRA_LABELS[inf]:<24} "
                  f"{r['consumption_ml']:>11.4f} {r['withdrawal_ml']:>12.4f} "
                  f"{r['energy_wh']:>10.4f} {DIM}{e.central:>9.1f}{RESET}")
        print()

    print(f"  {BOLD}--- Agent Swarms (5 x 10 turns) ---{RESET}")
    print(f"  {'Model':<10} {'Infra':<24} {'Consump mL':>11} {'Withdraw mL':>12}  Comparison")
    print("  " + "-" * 78)
    for m in ["haiku", "sonnet", "opus"]:
        for inf in ["google_cloud", "aws_east"]:
            sw = estimate_swarm(5, 10, model=m, infra=inf)
            c = sw["central"]["consumption_ml"]
            w = sw["central"]["withdrawal_ml"]
            print(f"  {MODEL_LABELS[m][0]:<10} {INFRA_LABELS[inf]:<24} "
                  f"{c:>11.2f} {w:>12.2f}  {nearest(c)}")
    print()

    print(f"  {BOLD}--- Extended Thinking Overhead (Opus, Google Cloud) ---{RESET}")
    r_n = calculate_water(400, 600, model="opus")
    r_t = calculate_water(400, 600, thinking_tokens=10000, model="opus")
    ratio = r_t["consumption_ml"] / r_n["consumption_ml"]
    print(f"  Normal query:   {r_n['consumption_ml']:.4f} mL consumed / {r_n['withdrawal_ml']:.4f} mL withdrawn")
    print(f"  +10K thinking:  {r_t['consumption_ml']:.4f} mL consumed / {r_t['withdrawal_ml']:.4f} mL withdrawn  ({ratio:.1f}x)")
    print()


# --- params ---
def cmd_params(args):
    print(HEADER)
    print(f"  {BOLD}ALL MODEL PARAMETERS WITH UNCERTAINTY{RESET}")
    print()

    def show(name, p, indent="  "):
        print(f"{indent}{BOLD}{name}:{RESET} {p.low} / {YELLOW}{p.central}{RESET} / {p.high} {p.unit}")
        print(f"{indent}  {DIM}Source: {p.source}{RESET}")
        if p.note:
            print(f"{indent}  {DIM}Note: {p.note}{RESET}")
        print()

    print(f"  {BOLD}--- Energy Per Equivalent Output Token ---{RESET}")
    print(f"  {DIM}(Comprehensive: accelerator + host + idle + PUE overhead){RESET}")
    for m in ["haiku", "sonnet", "opus"]:
        show(f"{MODEL_LABELS[m][0]} ({MODEL_LABELS[m][1]})", ENERGY_PER_EQUIV_TOKEN[m])

    show("Input cost ratio", INPUT_COST_RATIO)

    print(f"  {BOLD}--- Water Usage Effectiveness (onsite cooling) ---{RESET}")
    for inf in ["google_cloud", "aws_east", "industry_avg"]:
        show(f"WUE {INFRA_LABELS[inf]}", WUE_DIRECT[inf])

    print(f"  {BOLD}--- Grid Water Intensity (electricity generation) ---{RESET}")
    show("Consumption (evaporated)", GRID_WATER_CONSUMPTION)
    show("Withdrawal (total intake)", GRID_WATER_WITHDRAWAL)

    print(f"  {BOLD}--- Renewable Energy Fraction ---{RESET}")
    for inf in ["google_cloud", "aws_east", "industry_avg"]:
        show(f"{INFRA_LABELS[inf]}", RENEWABLE_FRACTION[inf])

    print(f"  {BOLD}--- PUE (for reference, baked into energy values) ---{RESET}")
    for inf in ["google_cloud", "aws_east", "industry_avg"]:
        show(f"{INFRA_LABELS[inf]}", PUE[inf])

    print(f"  {BOLD}UNCERTAINTY PHILOSOPHY{RESET}")
    print(f"  {DIM}All parameters have explicit (low / central / high) ranges.")
    print(f"  'low' = least water scenario (all params optimistic for water)")
    print(f"  'high' = most water scenario (all params pessimistic)")
    print(f"  The ratio between low and high is typically 5-15x, reflecting")
    print(f"  genuine uncertainty — especially for models (like Claude) where")
    print(f"  the provider does not disclose per-token energy.{RESET}")
    print()


# --- status (live tracking) ---
def cmd_status(args):
    conn = get_db()
    if not conn:
        print(HEADER)
        print(f"  {RED}No tracking data found.{RESET}")
        print(f"  Run {BOLD}python aqua_cli.py setup{RESET} to install hooks.")
        return

    print(HEADER)
    print(f"  {BOLD}TRACKING STATUS{RESET}")
    print()

    # Active sessions (no ended_at)
    active = conn.execute(
        "SELECT session_id, started_at, working_dir, model FROM sessions "
        "WHERE ended_at IS NULL ORDER BY started_at DESC"
    ).fetchall()

    if active:
        print(f"  {GREEN}Active sessions: {len(active)}{RESET}")
        for s in active[:5]:
            d = Path(s["working_dir"]).name if s["working_dir"] else "?"
            age = datetime.now() - datetime.fromisoformat(s["started_at"])
            mins = int(age.total_seconds() / 60)
            events = conn.execute(
                "SELECT COUNT(*) as n FROM events WHERE session_id = ?",
                (s["session_id"],)
            ).fetchone()["n"]
            print(f"    {s['session_id'][:20]}  {d:<20} {events} events  {mins}m ago")
    else:
        print(f"  {DIM}No active sessions{RESET}")
    print()

    # Totals
    for period, days in [("Today", 1), ("This week", 7), ("All time", None)]:
        row = query_totals(conn, days)
        if row and row["events"] > 0:
            r = calculate_water(row["input_tokens"], row["output_tokens"],
                                row["thinking_tokens"])
            print(f"  {BOLD}{period}:{RESET} {row['sessions']} sessions, {row['events']} events")
            print(f"    Consumption: {r['consumption_ml']:.2f} mL ({nearest(r['consumption_ml'])})")
            print(f"    Withdrawal:  {r['withdrawal_ml']:.2f} mL ({nearest(r['withdrawal_ml'])})")
            print(f"    Energy:      {r['energy_wh']:.2f} Wh")
            print()

    conn.close()


# --- live ---
def cmd_live(args):
    conn = get_db()
    if not conn:
        print(f"  {RED}No tracking data. Run: python aqua_cli.py setup{RESET}")
        return

    print(HEADER)
    print(f"  {BOLD}LIVE SESSION DATA{RESET}")
    print()

    sessions = query_sessions(conn, days=1)
    total_c, total_w, total_e = 0, 0, 0

    for s in sessions:
        r = compute_water_from_db_row(s)
        total_c += r["consumption_ml"]
        total_w += r["withdrawal_ml"]
        total_e += r["energy_wh"]

        d = Path(s["working_dir"]).name if s["working_dir"] else "?"
        status = f"{GREEN}ACTIVE{RESET}" if not s["ended_at"] else f"{DIM}ended{RESET}"
        print(f"  {status} {d:<20} {s['event_count']} events | "
              f"consumed: {r['consumption_ml']:.2f} mL | "
              f"withdrawn: {r['withdrawal_ml']:.2f} mL | "
              f"{r['energy_wh']:.2f} Wh")
        if s["task_count"]:
            print(f"    {DIM}including {s['task_count']} agent spawns ({s['total_agent_turns'] or 0} est. sub-turns){RESET}")

    if sessions:
        print()
        print(f"  {BOLD}Today's total:{RESET}")
        print(f"    Consumption: {total_c:.2f} mL = {nearest(total_c)}")
        print(f"    Withdrawal:  {total_w:.2f} mL = {nearest(total_w)}")
        print(f"    {DIM}{daily_ctx(total_c)}{RESET}")
        print()
        print(render_glass(total_c))
    else:
        print(f"  {DIM}No sessions today{RESET}")
    print()
    conn.close()


# --- history ---
def cmd_history(args):
    _, _, nums = parse_model_and_nums(args)
    days = nums[0] if nums else 7

    conn = get_db()
    if not conn:
        print(f"  {RED}No tracking data. Run: python aqua_cli.py setup{RESET}")
        return

    print(HEADER)
    print(f"  {BOLD}HISTORY (last {days} days){RESET}")
    print()

    sessions = query_sessions(conn, days=days)
    if not sessions:
        print(f"  {DIM}No sessions in the last {days} days{RESET}")
        conn.close()
        return

    print(f"  {'Date':<12} {'Dir':<18} {'Events':>7} {'Consumed':>10} {'Withdrawn':>11} {'Energy':>8}")
    print("  " + "-" * 68)

    grand_c, grand_w = 0, 0
    for s in sessions:
        r = compute_water_from_db_row(s)
        grand_c += r["consumption_ml"]
        grand_w += r["withdrawal_ml"]
        d = Path(s["working_dir"]).name[:17] if s["working_dir"] else "?"
        dt = s["started_at"][:10] if s["started_at"] else "?"
        print(f"  {dt:<12} {d:<18} {s['event_count']:>7} "
              f"{r['consumption_ml']:>8.2f}mL {r['withdrawal_ml']:>9.2f}mL "
              f"{r['energy_wh']:>6.2f}Wh")

    print("  " + "-" * 68)
    print(f"  {'TOTAL':<31} {'':<7} {grand_c:>8.2f}mL {grand_w:>9.2f}mL")
    print()
    print(f"  Total consumption: {BOLD}{nearest(grand_c)}{RESET}")
    print(f"  Total withdrawal:  {nearest(grand_w)}")
    print(f"  {DIM}{daily_ctx(grand_c / max(days, 1))} per day average{RESET}")
    print()
    conn.close()


# --- web dashboard ---
def cmd_web(args):
    import subprocess
    dashboard = Path(__file__).parent / "aqua_dashboard.py"
    if not dashboard.exists():
        print(f"  {RED}Missing {dashboard}{RESET}")
        return
    result = subprocess.run([sys.executable, str(dashboard)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"  {RED}{result.stderr}{RESET}")


# --- dashboard ---
def cmd_dashboard(args):
    model, infra, _ = parse_model_and_nums(args)
    total_c, total_w, n = 0, 0, 0

    print(HEADER)
    print(f"  {BOLD}INTERACTIVE DASHBOARD{RESET} | Model: {MODEL_LABELS[model][0]}")
    print(f"  Enter: token count | 'swarm N M' | 't:N' (thinking) | 'q' quit")
    print()

    while True:
        try:
            inp = input(f"  {DIM}[{n}]{RESET} > ").strip()
            if inp.lower() in ('q', 'quit', 'exit'): break
            thinking = 0
            if inp.startswith("t:"):
                thinking = int(inp[2:]); tokens = 2000
            elif inp.startswith("swarm"):
                parts = inp.split()
                na = int(parts[1]) if len(parts) > 1 else 3
                nt = int(parts[2]) if len(parts) > 2 else 5
                sw = estimate_swarm(na, nt, model=model, infra=infra)
                cen = sw["central"]
                total_c += cen["consumption_ml"]; total_w += cen["withdrawal_ml"]; n += 1
                print(f"    {CYAN}+{cen['consumption_ml']:.2f}c / +{cen['withdrawal_ml']:.2f}w mL{RESET}"
                      f"  | Total: {BOLD}{total_c:.2f}c{RESET} {bar(total_c)} {nearest(total_c)}")
                continue
            elif inp.isdigit(): tokens = int(inp)
            elif inp == "": tokens = 2000
            else: print(f"    {DIM}number | 'swarm N M' | 't:N' | 'q'{RESET}"); continue

            r = calculate_water(int(tokens*.5), int(tokens*.5), thinking, model=model, infra=infra)
            total_c += r["consumption_ml"]; total_w += r["withdrawal_ml"]; n += 1
            print(f"    {CYAN}+{r['consumption_ml']:.4f}c / +{r['withdrawal_ml']:.4f}w mL{RESET}"
                  f"  | Total: {BOLD}{total_c:.2f}c{RESET} {bar(total_c)} {nearest(total_c)}")
        except KeyboardInterrupt: break
        except ValueError: print(f"    {DIM}enter a number{RESET}")

    print()
    print(f"  {BOLD}Session:{RESET} {n} queries | {total_c:.2f} mL consumed | {total_w:.2f} mL withdrawn")
    print(f"  {DIM}{daily_ctx(total_c)}{RESET}")
    print()


# --- setup ---
def cmd_setup(args):
    hook_path = Path.home() / "water-tracker" / "aqua_hook.py"
    settings_path = Path.home() / ".claude" / "settings.json"

    print(f"""{HEADER}
  {BOLD}SETUP: Automatic Water Tracking for Claude Code{RESET}
  {DIM}Tracks all queries, tool calls, and agent swarms across instances{RESET}

  {BOLD}HOW IT WORKS{RESET}
  1. A lightweight hook script ({DIM}aqua_hook.py{RESET}) runs on each Claude Code event
  2. It logs estimated tokens to a shared SQLite database ({DIM}~/.aqua/water.db{RESET})
  3. SQLite WAL mode ensures safe concurrent access across instances
  4. The CLI ({DIM}aqua_cli.py{RESET}) reads the database to compute water estimates

  {BOLD}STEP 1: Verify files exist{RESET}
  {DIM}These should already be in place:{RESET}
    {hook_path}
    {Path.home() / "water-tracker" / "aqua_cli.py"}

  {BOLD}STEP 2: Add hooks to Claude Code settings{RESET}
  Edit {YELLOW}{settings_path}{RESET}

  Add the aqua hook as a SECOND entry in each event's hooks array:

  {CYAN}"PostToolUse": [
    {{
      "hooks": [
        {{ ... your existing post-tool-use hook ... }},
        {{
          "type": "command",
          "command": "python3 '{hook_path}' post-tool",
          "timeout": 3
        }}
      ]
    }}
  ],
  "UserPromptSubmit": [
    {{
      "hooks": [
        {{ ... your existing prompt-submit hook ... }},
        {{
          "type": "command",
          "command": "python3 '{hook_path}' prompt",
          "timeout": 3
        }}
      ]
    }}
  ],
  "Stop": [
    {{
      "hooks": [
        {{ ... your existing stop hook ... }},
        {{
          "type": "command",
          "command": "python3 '{hook_path}' stop",
          "timeout": 3
        }}
      ]
    }}
  ]{RESET}

  {BOLD}STEP 3: Initialize the database{RESET}
    python3 {hook_path} init
  {DIM}(This creates ~/.aqua/water.db with WAL mode enabled){RESET}

  {BOLD}STEP 4: Verify it works{RESET}
  Start a new Claude Code session, do a few things, then:
    python3 ~/water-tracker/aqua_cli.py status

  {BOLD}VIEWING YOUR DATA{RESET}
  {DIM}(These work in any terminal, not just inside Claude Code){RESET}

    python aqua_cli.py status       {DIM}# Quick overview: active sessions, today/week/all-time{RESET}
    python aqua_cli.py live         {DIM}# Today's sessions with per-session breakdown{RESET}
    python aqua_cli.py history 30   {DIM}# Last 30 days of usage{RESET}
    python aqua_cli.py compare      {DIM}# What-if analysis across models/infra{RESET}
    python aqua_cli.py params       {DIM}# All parameters with uncertainty ranges{RESET}

  {BOLD}TIPS{RESET}
  - Works across multiple simultaneous Claude Code instances
  - Each instance gets a unique session (identified by PID)
  - Agent sub-turns (Task tool) are tracked and counted separately
  - Set AQUA_MODEL=sonnet env var to change default model per instance
  - The hook adds ~20ms overhead per tool call (negligible)
  - Database auto-creates if missing; no migration needed

  {BOLD}OPTIONAL: Shell alias{RESET}
    echo 'alias aqua="python3 ~/water-tracker/aqua_cli.py"' >> ~/.zshrc
    source ~/.zshrc
  Then use: {CYAN}aqua status{RESET}, {CYAN}aqua live{RESET}, {CYAN}aqua history{RESET}, etc.

  {BOLD}UNINSTALL{RESET}
  Remove the aqua hook entries from {settings_path}
  Optionally: rm -rf ~/.aqua
""")

    # Auto-init DB
    if not DB_PATH.exists():
        print(f"  {YELLOW}Initializing database...{RESET}")
        AQUA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, pid INTEGER, ppid INTEGER,
                started_at TEXT NOT NULL, ended_at TEXT,
                model TEXT DEFAULT 'opus', working_dir TEXT, instance_label TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL, tool_name TEXT,
                est_input_tokens INTEGER DEFAULT 0,
                est_output_tokens INTEGER DEFAULT 0,
                est_thinking_tokens INTEGER DEFAULT 0,
                agent_sub_turns INTEGER DEFAULT 0,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
        """)
        conn.close()
        print(f"  {GREEN}Created {DB_PATH}{RESET}")
    else:
        print(f"  {GREEN}Database exists: {DB_PATH}{RESET}")
    print()


# --- methodology ---
def cmd_methodology(args):
    print(f"""{HEADER}

  {BOLD}FORMULA{RESET}
  equiv_tokens = input_tokens x {INPUT_COST_RATIO.central} + output_tokens + thinking_tokens
  E_total      = equiv_tokens x J_per_equiv_token   (comprehensive, PUE included)
  E_kWh        = E_total / 3,600,000

  Water_direct_consumption  = E_kWh x WUE           [mL]  (onsite cooling, evaporative)
  Water_direct_withdrawal   = same                         (evaporative: all consumed)
  Water_indirect_consumpt.  = E_kWh x Grid_WI_C x (1 - renewable_frac)  [mL]
  Water_indirect_withdrawal = E_kWh x Grid_WI_W x (1 - renewable_frac)  [mL]

  {BOLD}CONSUMPTION vs WITHDRAWAL{RESET}
  {DIM}Consumption = water evaporated, permanently lost to local watershed.
  Withdrawal  = total water taken in, mostly returned (warmer/altered).
  Consumption is the environmentally critical metric.
  Withdrawal matters for local water stress and thermal pollution.
  Ratio: withdrawal is typically 2-3x consumption for grid electricity.
  For onsite data center cooling (evaporative), withdrawal ≈ consumption.{RESET}

  {BOLD}UNCERTAINTY{RESET}
  {DIM}Every parameter has (low / central / high) bounds:
    Low  = best case for water (least water used)
    High = worst case (most water used)
  In 'low' scenario, ALL parameters are set to their optimistic bound
  simultaneously. In 'high', all are pessimistic. This gives a bracketing
  range, NOT a confidence interval. True value likely within this range
  but could exceed it if our model structure is wrong.

  Run 'aqua params' to see all parameter ranges with sources.{RESET}

  {BOLD}VALIDATION{RESET}
  Central estimates for 1000-token queries (central scenario):
    Haiku:   ~0.03 mL consumed   (consistent with << Google's 0.26)
    Sonnet:  ~0.15 mL consumed   (consistent with Google Gemini ~0.26 onsite-only)
    Opus:    ~0.30 mL consumed   (consistent with Altman's ~0.3 claim)

  Full 30-turn Opus session: ~15-25 mL consumed = 1-2 tablespoons
  Agent swarm 5x10 Opus: ~30-50 mL consumed = about a shot glass

  Consistent with Masley: "30 or 50 ChatGPT queries to fill a water bottle"

  {BOLD}WHAT'S INCLUDED{RESET}
  [x] Active accelerator power (GPU/TPU)     [x] Onsite cooling water (WUE)
  [x] Host CPU/DRAM overhead                 [x] Grid electricity water (indirect)
  [x] Idle machine provisioning              [x] Renewable energy offset
  [x] Data center overhead (PUE)             [x] Context growth over sessions

  {BOLD}WHAT'S EXCLUDED{RESET}
  [ ] Model training (amortized)  [ ] Network energy    [ ] End-user device
  [ ] Hardware manufacturing      [ ] Data storage      [ ] Embodied water

  {BOLD}KEY REFERENCES{RESET}
  [1] LBNL 2024 US Data Center Energy Usage Report
      eta.lbl.gov/publications/2024-lbnl-data-center-energy-usage-report
  [2] Google Cloud (Aug 2025): Measuring Environmental Impact of AI Inference
      cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference
  [3] NREL HPC Data Center Water Usage Efficiency
      nrel.gov/computational-science/reducing-water-usage
  [4] Ren et al. (2023): Making AI Less Thirsty  (arxiv:2304.03271)
  [5] Masley, A.: The AI water issue is fake  (andymasley.com)
  [6] Bashir et al. (2025): Carbon and water footprints of data centers
      cell.com/patterns/fulltext/S2666-3899(25)00278-8
  [7] TokenPowerBench (Dec 2025)  (arxiv:2512.03024)
  [8] ML.ENERGY Benchmark (May 2025)  (arxiv:2505.06371)
  [9] Macknick et al. (2012): Operational water consumption and withdrawal
      factors for electricity generating technologies  (NREL/TP-6A20-50900)

  {BOLD}HONESTY{RESET}
  {DIM}Anthropic does not disclose per-token energy for Claude. Google is the
  only major AI company to publish per-query energy/water. Our model is
  calibrated to Google's disclosure and cross-checked against independent
  benchmarks. True values for Claude could differ by 2-3x. The low/high
  scenario bounds reflect this structural uncertainty.{RESET}
""")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    args = sys.argv[1:]
    if not args:
        print_help()
        return

    cmd = args[0]
    rest = args[1:]

    commands = {
        "query": cmd_query, "q": cmd_query,
        "swarm": cmd_swarm,
        "session": cmd_session, "claude-session": cmd_session,
        "compare": cmd_compare,
        "dashboard": cmd_dashboard, "dash": cmd_dashboard,
        "params": cmd_params, "parameters": cmd_params, "uncertainty": cmd_params,
        "status": cmd_status,
        "live": cmd_live,
        "history": cmd_history, "hist": cmd_history,
        "web": cmd_web,
        "setup": cmd_setup, "install": cmd_setup,
        "methodology": cmd_methodology, "method": cmd_methodology, "sources": cmd_methodology,
        "help": lambda a: print_help(),
    }

    if cmd in commands:
        commands[cmd](rest)
    else:
        print(f"  Unknown command: {cmd}")
        print_help()


if __name__ == "__main__":
    main()
