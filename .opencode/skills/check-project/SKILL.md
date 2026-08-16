---
name: check-project
description: Check a project located at ~/code/<name>. Use when the user asks to check, review, inspect, or look at a project by name — e.g. "I want you to check the my-app project". Resolves the project name to ~/code/<name>, explores its structure, assesses health (git status, tests, lint), and reports a summary. Do NOT use for the current project (story-engine); use only for other projects in ~/code/.
---

# Check a Project

Triggered by requests like "I want you to check the <name> project",
"check <name>", or "review/inspect <name>". The project is always
located at `~/code/<name>` — the folder name is exactly the name the
user gave.

## 1. Resolve the path

The project lives at:

```bash
~/code/<project-name>
```

- Use the exact name the user said (e.g. "my-app" →
  `~/code/my-app`).
- Verify the directory exists. If it does not, list `~/code/` to find
  close matches and ask the user which project they meant — do not
  guess a different name silently.

## 2. Explore the project

Build an understanding of what the project is before judging it:

1. Read the top-level files: `README`, `AGENTS.md` (or similar agent
   instructions), any `docs/` directory, and the project's own
   documentation.
2. Identify the tech stack from manifest files: `Makefile`,
   `pyproject.toml`, `requirements.txt`, `package.json`, `Cargo.toml`,
   `go.mod`, `composer.json`, etc.
3. Map the structure: main source directories, entry points, tests,
   and how the project is built/run/tested.
4. Read the key files that define behavior — the entry point, core
   service, or main module — not just the manifests.

## 3. Assess project health

1. Check git state: `git status --short` and `git log --oneline -5`
   (what branch, how many uncommitted changes, recent activity).
2. Run the project's own checks if present — always prefer the
   project's own commands over generic ones (e.g. `make test`,
   `make lint`, `npm test`). If the project has no such commands,
   note that instead of running arbitrary commands.
3. Look for obvious problems: failing tests, lint violations, missing
   or stale docs, dead code, TODOs/FIXMEs in critical paths,
   uncommitted work, untracked files that look important.

Do NOT modify anything. This is a read-only inspection.

## 4. Report

Give the user a concise structured summary covering:

- **What it is** — purpose, stack, key entry points
- **Structure** — main directories/modules and their roles
- **Health** — git state, test/lint results, any problems found
- **Notable observations** — things worth the user's attention (risks,
  suspicious code, outdated dependencies, missing tests, etc.)

Keep it tight: bullet points, no essay. End with a short
"anything else" prompt so the user can tell you what to dig into
next.
