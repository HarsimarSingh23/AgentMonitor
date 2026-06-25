# contextdb

An **in-memory observability database for LLM agents.** Drop decorators on your
tools and a few one-line loggers in your agent loop, and contextdb records the
whole story of a run on a timeline:

- every **tool call** (args, return value, latency, errors)
- every **user message** and **assistant response**
- the agent's **thinking / reasoning**
- **world changes** the agent made (file edits, DB writes, …)
- **token counts** for everything, and — the part most tools miss —
- **what context was carried in from previous turns/conversations/memory**, and
  how many tokens it cost.

Then it gives you **insights** over that log so you can answer: *how is the agent
actually working, and how should I set up context for the next conversation?*

Zero required dependencies. Uses `tiktoken` for accurate token counts if it's
installed, otherwise a heuristic.

## Install

```bash
pip install -e .            # core
pip install -e ".[tokens]"  # + accurate token counting via tiktoken
```

## Quickstart

```python
import contextdb as cdb

@cdb.log_tool                 # every call to this tool is now recorded
def web_search(query: str) -> list[str]:
    ...

with cdb.session(parent_session_id=last_run_id) as s:
    s.log_system_prompt("You are a research agent.")

    # The key feature: declare what context you re-fed into this turn.
    s.log_context([
        cdb.ContextRef("previous_session", last_run_id, "summary of prior chat", tokens=180),
        cdb.ContextRef("memory", "mem-7", "user prefers concise answers", tokens=20),
    ])

    s.log_user_message("Find docs about HTTP retries.")
    with s.turn("agent_turn"):          # nest tool calls under one turn
        s.log_thinking("I'll search the web first.")
        web_search("HTTP retries")
    s.log_response("Found 3 results.")
    s.log_change("wrote report.md", target="report.md")

ins = cdb.insights()
print(ins.session_report(s.id).to_dict())   # tokens, tool stats, carryover ratio
print(ins.context_flow(s.id))               # exactly what context was carried in
print(ins.token_timeline(s.id))             # cumulative token growth per event
```

## What you get

`Insights.session_report(session_id)` returns, per run:

- `by_type` — counts of each event kind
- `input_tokens` / `output_tokens` / `total_tokens`
- `carried_context_tokens` and **`context_carryover_ratio`** — the share of your
  input budget spent re-feeding old context vs. the user's new input
- `context_sources` — token cost broken down by where context came from
- `tool_stats` — calls, error rate, avg latency, tokens per tool
- `errors`, `duration_ms`

Other queries: `context_flow`, `token_timeline`, `tool_leaderboard`, and a
store-wide `summary`.

## Live dashboard

Watch your agent work in real time. `cdb.serve()` starts a tiny stdlib HTTP
server that streams every event to the browser over Server-Sent Events the
instant it's logged — no build step, no JS dependencies.

```python
import contextdb as cdb

cdb.serve()          # non-blocking; opens http://127.0.0.1:8765 in your browser

with cdb.session() as s:
    s.log_user_message("…")     # each event pops into the feed live
    ...
```

The dashboard shows a streaming event feed (color-coded by type, with i/o,
latency, token counts, and errors), running header stats (tokens in/out, **live
context-carryover %**, tool calls, errors), per-type filter chips, and a session
selector. Context injections are highlighted with their source breakdown so you
can see exactly what was re-fed from prior conversations.

Try the full live simulation:

```bash
python -m examples.live_demo     # launches the dashboard + a fake agent run
```

Options: `cdb.serve(port=8765, host="127.0.0.1", open_browser=True, block=False)`.
Use `block=True` to serve in the foreground.

## Live control plane — drive the agent from the dashboard

Monitoring is one-directional. The control plane adds the return channel so you
can **steer a running agent** from the browser. Commands POST to the server,
mutate a `ControlPlane`, and the agent **checks in** at checkpoints our
decorators and `session.turn()` already provide.

> **Constraint:** you can't forcibly freeze arbitrary Python from outside the
> process — the agent only stops where it checks in. Instrumented code
> (`@log_tool` + `turn()`) checks in automatically. For hand-written loops, drop
> `cdb.checkpoint("label")` wherever you want to be interruptible.

From the dashboard control bar you can:

| Action | What it does |
|---|---|
| **Pause / Resume / Step** | Gate the agent at the next checkpoint; step advances one checkpoint then re-pauses |
| **Breakpoint** | Auto-pause right *before* a named tool runs |
| **Patch a tool → force return** | Short-circuit a tool with a fixed value — *change the scenario* without touching code |
| **Patch a tool → inject error** | Make a tool raise, to test the agent's error handling |
| **Inject context** | Queue a `ContextRef` that lands in the agent's **next turn** |
| **Abort** | Raise `AgentAborted` inside the agent at its next checkpoint |

Live demo (loops an agent so you can poke it):

```bash
python -m examples.control_demo
```

It's also a plain HTTP/JSON API, so you can script it:

```python
import requests
requests.post("http://127.0.0.1:8765/control", json={"cmd": "pause"})
requests.post("http://127.0.0.1:8765/control",
              json={"cmd": "patch_tool", "name": "db.query", "action": "return", "value": "{\"rows\": 0}"})
requests.post("http://127.0.0.1:8765/control",
              json={"cmd": "queue_context", "source_kind": "ops", "summary": "DB is degraded", "tokens": 40})
requests.post("http://127.0.0.1:8765/control", json={"cmd": "resume"})
```

…or from Python directly via `cdb.get_control()` (`pause()`, `step()`,
`patch_tool()`, `set_breakpoint()`, `queue_context()`, `abort()`).

## Capturing real agents (adapters)

`@log_tool` covers code you write. To observe *real* agents, `contextdb.adapters`
plugs into three sources. A note on reach: a model's hidden chain-of-thought
only leaves the provider when you ask for it — so **reasoning capture comes from
the Anthropic SDK with extended thinking**, while Claude Code hooks/transcripts
give you tool calls, results, prompts, and token usage.

### 1. Anthropic SDK — logs the agent's actual thinking

```python
import anthropic, contextdb as cdb
from contextdb.adapters.anthropic_sdk import log_response

client = anthropic.Anthropic()
with cdb.session() as s:
    s.log_user_message(prompt)
    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=1024,
        thinking={"type": "enabled", "budget_tokens": 2048},
        messages=[{"role": "user", "content": prompt}],
    )
    log_response(s, resp)   # thinking blocks → THINKING, tool_use → TOOL_CALL, text → response
```

`log_response` / `stream_log` handle non-streaming and streaming responses;
`mock_response()` lets the demo run with **no API key**:

```bash
python -m examples.anthropic_demo
```

### 2. Claude Code hooks — watch a live Claude Code agent

Start the collector, register the hook, and work as usual:

```bash
python -m contextdb        # collector + dashboard at http://127.0.0.1:8765
```

Add to `.claude/settings.json` (or get it from `claude_code_hooks.settings_snippet()`):

```json
{
  "hooks": {
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python -m contextdb.adapters.claude_code_hooks"}]}],
    "PostToolUse":      [{"hooks": [{"type": "command", "command": "python -m contextdb.adapters.claude_code_hooks"}]}],
    "Stop":             [{"hooks": [{"type": "command", "command": "python -m contextdb.adapters.claude_code_hooks"}]}]
  }
}
```

Each hook ships its payload to the collector's `/ingest` endpoint. The hook
never raises into Claude Code — if the collector is down it exits cleanly.

### 3. Transcript import — analyze past sessions

```bash
python -m contextdb.adapters.transcript_import ~/.claude/projects/<slug>/<session>.jsonl
```

Replays a recorded session (`text` / `thinking` / `tool_use` blocks, tool
results, token usage) into a store and prints the `Insights` report.

## Persistence

The store is in-memory, but you can snapshot it for later analysis:

```python
cdb.get_store().to_json("run.json")
cdb.get_store().to_sqlite("run.sqlite")   # then query with plain SQL
```

## Architecture

| Module | Responsibility |
|---|---|
| `models.py` | `Event`, `EventType`, `ContextRef`, `TokenUsage` |
| `tokens.py` | token counting (tiktoken or heuristic) |
| `store.py` | thread-safe in-memory store + subscribers + JSON/SQLite snapshots |
| `session.py` | `Session`, turns, ambient "current session" via contextvars |
| `decorators.py` | `@log_tool`, `@log_function`, default store |
| `insights.py` | analytics over the timeline |
| `dashboard.py` + `static/dashboard.html` | live SSE monitoring dashboard + control endpoints |
| `control.py` | live control plane: pause/step, breakpoints, tool patches, context injection, abort |
| `adapters/` | feed real agents in: Anthropic SDK (incl. thinking), Claude Code hooks, transcript import |

## Run the demo / tests

```bash
python -m examples.demo
python -m pytest -q
```
