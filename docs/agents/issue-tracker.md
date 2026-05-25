# Issue tracker: Local markdown

Issues and PRDs for this repo live as markdown files under `.scratch/`.

## Conventions

- **Create an issue**: Create a directory `.scratch/<feature-slug>/` and write an `issue.md` inside it.
- **Read an issue**: Read `.scratch/<feature-slug>/issue.md`.
- **List issues**: List directories under `.scratch/`.
- **Comment on an issue**: Append to `.scratch/<feature-slug>/issue.md` or create a `comments.md` alongside it.
- **Apply / remove labels**: Add or remove a `labels:` frontmatter field in `issue.md` (comma-separated).
- **Close**: Move the directory to `.scratch/_closed/<feature-slug>/` or add a `status: closed` frontmatter field.

## When a skill says "publish to the issue tracker"

Create a `.scratch/<feature-slug>/issue.md` file.

## When a skill says "fetch the relevant ticket"

Read `.scratch/<feature-slug>/issue.md`.