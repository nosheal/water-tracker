# AQUA — AI Query Water Usage Analyzer

Track the water footprint of your AI coding sessions.

AQUA hooks into [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and estimates the water consumed and withdrawn by each interaction — datacenter cooling, grid electricity generation, and upstream supply chain — then renders a retro CRT dashboard you can keep open in a browser tab.

![Dashboard](https://img.shields.io/badge/dashboard-retro%20CRT-00ffff?style=flat-square)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

## How it works

```
Claude Code ──hook──▶ aqua_hook.py ──▶ ~/.aqua/water.db (SQLite)
                           │
                           ▼ (every 15s, background)
                     aqua_dashboard.py ──▶ ~/.aqua/dashboard.html
                                                │
                                        browser auto-refreshes
```

1. **`aqua_hook.py`** fires on every tool call, prompt, and session stop. It estimates token counts per tool type and logs them to a local SQLite database (`~/.aqua/water.db`) in <50ms.
2. **`aqua_dashboard.py`** queries the database, computes water estimates across six time periods (today / yesterday / this week / this month / this year / all time), and bakes everything into a single self-contained HTML file.
3. **`aqua_dashboard.html`** is the dashboard template — no server required. The browser auto-refreshes every 30 seconds to pick up new data.

## Technoeconomic model

Calibrated to Google's August 2025 Gemini disclosure (0.24 Wh, 0.26 mL per median prompt).

| Parameter | Low | Central | High | Source |
|-----------|-----|---------|------|--------|
| Opus energy (J/equiv token) | 1.5 | 4.0 | 10.0 | Scaled from Gemini disclosure |
| Sonnet energy | 0.8 | 2.0 | 5.0 | |
| Haiku energy | 0.15 | 0.4 | 1.0 | |
| Datacenter WUE (L/kWh) | 0.5 | 1.08 | 1.8 | Google Environmental Report 2024 |
| Grid water consumption (L/kWh) | 0.8 | 1.8 | 2.7 | Macknick et al. (2012) |
| Grid water withdrawal (L/kWh) | 2.0 | 4.5 | 7.0 | Macknick et al. (2012) |
| Renewable fraction | 64% | 50% | 30% | Ember Global Electricity Review 2024 |

**Water = direct cooling + indirect grid water × (1 − renewable fraction)**

Input tokens are weighted at 25% of output tokens (cheaper to process than generate).

## Quick reference

| Scenario | Water per 50-turn Opus session |
|----------|-------------------------------|
| Low | ~60 mL (a shot glass) |
| Central | ~600 mL (a water bottle) |
| High | ~3,000 mL (a wine bottle+) |

## Installation

### 1. Clone

```bash
git clone https://github.com/nosheal/water-tracker.git ~/water-tracker
```

### 2. Configure Claude Code hooks

Add to your Claude Code settings (`~/.claude/settings.json`):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/water-tracker/aqua_hook.py post-tool"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/water-tracker/aqua_hook.py prompt"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/water-tracker/aqua_hook.py stop"
          }
        ]
      }
    ]
  }
}
```

### 3. Open the dashboard

```bash
python3 ~/water-tracker/aqua_dashboard.py
```

This generates `~/.aqua/dashboard.html` and opens it in your browser. Leave the tab open — it auto-refreshes every 30 seconds as the hook regenerates the file in the background.

## Dashboard features

- **Water beaker** — animated fill level scaled to one toilet flush (6L) for perspective
- **Draining globe** — Earth visualization where water recedes and mountain peaks emerge, driven by a real-time estimate of global AI water consumption
- **Interactive uncertainty slider** — drag between LOW and HIGH scenarios; all metrics, benchmarks, and visualizations update live
- **Distribution slider** — vertical control to explore "what if everyone used AI more/less than me?" (0.1× to 10×)
- **Benchmarks** — toilet flushes, shower seconds, glasses of water, raindrops, bottles, ice cubes, faucet time, % of daily household use
- **Consumed / Withdrawn toggle** — click either metric to switch the benchmark comparisons
- **Time periods** — TODAY, YESTERDAY, THIS WEEK, THIS MONTH, THIS YEAR, ALL TIME
- **Session list** — active/ended sessions with working directory, event count, model, water usage
- **Accumulation chart** — cumulative water over time
- **Tool breakdown** — which tools use the most water
- **Token summary** — raw input/output/thinking/agent token counts
- **7-day history** — daily water usage bar chart
- **Live refresh** — auto-reloads every 30s; manual refresh button in header

## CLI usage

```bash
# Full CLI with history, comparisons, and export
python3 ~/water-tracker/aqua_cli.py

# Dashboard only
python3 ~/water-tracker/aqua_dashboard.py        # open in browser
python3 ~/water-tracker/aqua_dashboard.py --out   # print file path only
```

## Global AI water estimate

The dashboard includes a real-time rolling estimate of worldwide AI water consumption, built from two independent approaches:

**Bottom-up (user distribution model):**

| Segment | Users | Queries/day | Model tier | mL/query |
|---------|-------|-------------|------------|----------|
| Casual | 480M | 3 | small | 0.04 |
| Regular | 200M | 15 | medium | 0.40 |
| Power | 80M | 50 | mixed | 1.24 |
| Agentic | 32M | 200 | large | 2.30 |
| Enterprise | 8M eq. | 100 | medium | 0.58 |

Subtotal ×8 (non-chatbot AI) ×2.5 (infrastructure overhead) ≈ **440M L/day**

**Top-down (IEA energy projection):**

IEA Electricity 2024 projects 100–150 TWh/yr AI datacenter energy → 540–810M L/day

Central convergence: **~600M L/day ≈ 7,000 L/sec**

## Data storage

All data is local:

```
~/.aqua/
├── water.db          # SQLite database (WAL mode)
└── dashboard.html    # Generated dashboard (self-contained)
```

No data is sent anywhere. The hook reads stdin from Claude Code's hook system and writes only to the local database.

## Sources

- [IEA Electricity 2024](https://www.iea.org/reports/electricity-2024) — AI datacenter energy projections
- [Google Environmental Report 2024](https://sustainability.google/reports/google-2024-environmental-report/) — WUE and per-query water data
- Macknick et al. (2012), *Operational water consumption and withdrawal factors for electricity generating technologies*
- [Ember Global Electricity Review 2024](https://ember-climate.org/insights/research/global-electricity-review-2024/) — Renewable energy share by region

## License

MIT
