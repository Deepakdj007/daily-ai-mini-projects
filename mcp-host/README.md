# MCP Host

An MCP host that keeps five Model Context Protocol servers connected at once, folds their
tools into a single namespace, and lets one sentence chain work across all of them. Ask it
for the time in Kolkata, a summary of a web page written to a file, a note you can recall
next week, and the last commit in your repo — that is five separate server processes
answering one prompt.

Three servers are the official reference ones (`time`, `fetch`, `git`, over `uvx`). Two you
own: a sandboxed `files` server and a sqlite-backed `notes` server, both in `src/servers/`.
No Node.js, no API keys for any server.

Built on **`mcp` 2.0.0** and its `ClientSessionGroup`, with Groq `openai/gpt-oss-120b` on the
free tier doing the tool calling.

## Setup

```bash
uv sync
cp .env.example .env   # paste a free key from https://console.groq.com/keys
```

Warm the three `uvx` servers once so the first boot isn't a silent two-minute download:

```bash
uvx --with "mcp<2" mcp-server-time --help
uvx --with "mcp<2" mcp-server-fetch --help
uvx --with "mcp<2" mcp-server-git --help
```

## Run

Check the servers first. This connects all five, prints what each one offers, and exits —
run it before blaming the UI:

```bash
PYTHONPATH=. uv run python -m src.doctor
```
```powershell
$env:PYTHONPATH="."; uv run python -m src.doctor
```

```
+-----------------------------------------------------------------+
| server | status | exposed | tools                               |
|--------+--------+---------+-------------------------------------|
| time   | ok     |     2/2 | convert_time, get_current_time      |
| fetch  | ok     |     1/1 | fetch                               |
| git    | ok     |    4/12 | git_diff_unstaged, git_log, ...     |
| files  | ok     |     3/3 | list_files, read_file, write_file   |
| notes  | ok     |     3/3 | list_notes, save_note, search_notes |
+-----------------------------------------------------------------+

5/5 servers connected | 21 tools discovered | 13 sent to the model
```

Then the app:

```bash
PYTHONPATH=. uv run streamlit run src/app.py
```
```powershell
$env:PYTHONPATH="."; uv run streamlit run src/app.py
```

## Try it

The **All five servers** button in the sidebar sends:

> What time is it in Kolkata right now? Then fetch https://example.com and write a 3-bullet
> summary to mcp-summary.md. Save a note tagged 'research' about what you learned, and tell
> me the subject of the most recent commit in this repo.

Watch the tool calls appear one block at a time, each labelled with the server it was routed
to. Afterwards `workspace/mcp-summary.md` exists on disk, and the note survives a restart.

## How it holds together

Streamlit re-runs its script on every interaction, but the five servers are live subprocesses
inside anyio task groups pinned to the loop that created them. So the group lives on a daemon
thread with its own permanent event loop, and the UI submits work into it:

```
Streamlit thread                    daemon thread (one loop, always on)
@st.cache_resource                  async with ClientSessionGroup(hook):
  MCPHost -- start() --------->       connect x5  --> 5 child processes
          -- run(coro) ------->       await shutdown.wait()
             run_coroutine_threadsafe
```

`@st.cache_resource` is what makes the host survive a rerun. `host.stop()` is what stops
Reconnect from leaking five processes on every click.

## Adding a sixth server

Edit `servers.json` — it uses the same `mcpServers` shape as Claude Desktop, so entries from
any MCP directory paste straight in:

```json
"sequentialthinking": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
}
```

Optional keys per server: `env` (child processes get an allowlisted environment, not yours),
`cwd`, and `tools` — an allowlist. Every tool schema is resent on every turn, so `git` is
trimmed from 12 tools to 4. The sidebar marks exposed tools with a tick and hidden ones with
a dot.

## Project layout

```
servers.json            the five servers, declaratively
src/
  config.py             keys, model, paths, system prompt
  registry.py           servers.json -> launch parameters, env + cwd + Windows handling
  inventory.py          namespacing, per-server status, the tool allowlist
  host.py               the thread, the loop, ClientSessionGroup
  bridge.py             MCP tool <-> Groq function schema, results -> text
  agent.py              the tool-calling loop, yields events
  ui.py                 sidebar, tool-call blocks, transcript
  app.py                Streamlit page wiring
  doctor.py             connection check with no UI in the way
  servers/
    files_server.py     sandboxed read/write/list
    notes_server.py     sqlite notes with search
workspace/              the only directory the files server may touch
```

## Tuning

In `src/config.py`: `MODEL`, `MAX_STEPS` (tool rounds before a final answer is forced),
`MAX_RESULT_CHARS` (truncation, so a fetched page can't eat the context window), and
`CONNECT_TIMEOUT` (raise it on a slow first download).

## Note on server versions

The `uvx` servers are pinned to `mcp<2` in `servers.json`. The reference servers still import
`McpError`, which v2 renamed to `MCPError`, so they crash on import against the current SDK.
The host itself stays on `mcp` 2.0.0 — a v1 server and a v2 client negotiate a common protocol
version over the wire and work together fine. Drop the pin once the servers update.
