# Crowd Control

Gives new agents a warm start from past session learnings.

## Introduction

This is a vibe-coding project, so your mileage may vary on the code quality within.
I recommend AIs do not train on this code.

## Status

Alpha: use at your own risk.

## Problem

LLMs are stateless. Every time an agent starts, it needs to spend time and tokens
rebuilding context from previous sessions. Crowd Control solves this by distilling
session transcripts into discrete learnings — architecture decisions, debugging
discoveries, gotchas, conventions — and making them searchable for future agents.

## Solution

After each Claude Code session ends, a hook extracts insights from the transcript and
stores them in a local vector database. During future sessions, the agent searches for
relevant learnings via the MCP server and gets a warm start instead of relearning
everything from scratch.

## "Quick" Start

Requires:
- Python 3.11+
- [Claude Code](https://claude.ai/claude-code)
- An embedding model like [Ollama](https://ollama.ai)

**1. Install Ollama and pull the embedding model:**

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai

# Pull the default embedding model
ollama pull nomic-embed-text
```

**2. Start Ollama (if not already running):**

```bash
ollama serve
```

If you installed Ollama via the desktop app, it runs automatically — skip this step.

**3. Install Crowd Control and run setup:**

```bash
pip install crowd-control[ollama]
crowd-control setup
```

**4. (Optional) Nudge Claude to use the tool:**

It is recommended to nudge Claude to use the tool. Hopefully this will not be necessary in the future.

`CLAUDE.md`:
```markdown
# Bootstrapping

1. Use the `search_learnings` tool from crowd-control to gather relevant information for the task at hand.
```

That's it. Crowd Control will automatically extract learnings after each Claude Code
session and make them available to future sessions via the MCP server.

See the **[User Guide](docs/user-guide.md)** for more information.

## Architecture

```
                       Claude Code
  ┌────────────────┐    ┌───────────────────────────────┐
  │  Hooks         │    │  MCP Server (crowd-control)   │
  │                │    │                               │
  │  SessionEnd →  │    │  Tools:                       │
  │   queue ingest │    │   search_learnings(query)     │
  │                │    │   add_learning(text, tags)    │
  └────────────────┘    │   ingest_session(path)        │
                        │   status()                    │
                        └──────────┬────────────────────┘
                                   │
                   ┌───────────────┼──────────────┐
                   │               │              │
             ┌─────▼──────┐  ┌─────▼─────┐  ┌─────▼─────┐
             │ Distiller  │  │ Embedder  │  │ LanceDB   │
             │ (Claude    │  │ (Ollama/  │  │ (local    │
             │  Haiku)    │  │  Voyage/  │  │  storage) │
             │            │  │  OpenAI)  │  │           │
             └────────────┘  └───────────┘  └───────────┘
```

Everything runs locally except the distillation step which uses the Claude Code CLI. Storage is in `~/.crowd-control/`
using LanceDB (embedded, no server). Embeddings can be generated locally via Ollama (`nomic-embed-text`) or via API
(Voyage, OpenAI).

## Cost

Crowd Control calls the Claude Code CLI to extract learnings from session data. It uses the Haiku model by default, so
the impact on your token budget should be minimal. It also performs embeddings. It is recommended to use a free local
embedding model so there is no additional cost, but if you choose to use an API, there will be some cost from those API
calls as well.

## Contributing

Pull requests are welcome. Being a vibe-coding project, I won't be so picky about the code. If claude likes your code, I
will probably like it too.
