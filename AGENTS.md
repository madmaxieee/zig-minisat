## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

> **Important:** `.scratch/` is gitignored. Builtin tools (glob, grep, fff MCP, etc.) respect `.gitignore` and will **not** find issue files. Use shell commands to `grep` for issue files, or use the `read` tool with explicit paths:

### Triage labels

Uses the default five-role vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Issue-fixing workflow

For each issue, after applying the fix:

1. **Format** — Run `zig fmt` on the modified file(s).
2. **Test** — Run `zig build test` to verify existing tests still pass.
3. **Commit** — Use `jj describe -m "type(scope): description"` with a conventional commit message. This describes the *current* working copy — do **not** run `jj new` first.
4. **Update issue status** — Rewrite the issue file: change `labels: ready-for-agent` to `labels: done`, and check off completed acceptance criteria with `[x]`.
5. **Next issue** — Run `jj new` to create a new revision for the next issue. This must come **after** `jj describe`, not before.
