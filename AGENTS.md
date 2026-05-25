## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

> **Important:** `.scratch/` is gitignored. Builtin tools (glob, grep, fff MCP, etc.) respect `.gitignore` and will **not** find issue files. Use shell commands to `grep` for issue files, or use the `read` tool with explicit paths:

### Triage labels

Uses the default five-role vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
