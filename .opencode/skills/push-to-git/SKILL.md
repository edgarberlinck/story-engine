---
name: push-to-git
description: Push the current work to git after verifying it is clean. Runs a syntax check, lint, and the full test suite in that order; if all pass, commits ALL changes and pushes to the remote. Use when the user says "push to git", "send this to git", "push it", "commit and push", "release this", or any similar request to ship the current work to the repository. Do not use for merely inspecting a project (use check-project) or when the user explicitly wants to skip the checks.
---

# Push to Git

Run every step below, in order. If any step fails, **STOP and report the
failure to the user — do not commit or push** until the failing check is
resolved. Do not skip a step unless the user explicitly asks. Prefer the
project's own commands (from the `Makefile`) over generic ones.

This repo uses a virtualenv at `.venv` (Python 3.14). All tool invocations
below use it, e.g. `.venv/bin/python`. If `.venv` does not exist, run
`make install` first.

## 1. Syntax check must pass

Every module must byte-compile cleanly, catching any syntax errors before
anything else:

```bash
.venv/bin/python -m compileall -q .
```

Any compile error fails this step. Fix the flagged file(s) and re-run until
clean. (If a real build/packaging system is added later — `pyproject.toml`,
wheels, etc. — replace this with the actual build command.)

## 2. Lint must pass

Run the project's linter:

```bash
make lint
```

This runs `flake8 .`. Fix any reported violations before continuing. Prefer
`make format` (black) to auto-fix style issues, then re-run lint.

Note: `flake8 .` scans `.venv` as well, so a RecursionError from third-party
packages can be reported instead of the project's own violations. If that
happens, run flake8 scoped to the project source instead:

```bash
.venv/bin/flake8 --exclude=.venv,outputs,node_modules,__pycache__ .
```

Do not treat a third-party `.venv` crash as a success — scope the run so the
project's own code is actually linted, then continue.

## 3. Tests must pass

Run the full test suite:

```bash
make test
```

This runs `python -m unittest discover -s . -p "test_*.py" -v`. If it reports
anything other than `OK`, fix the failures (or report them) and re-run until
green.

## 4. Commit ALL changes and push

All three checks now pass. Inspect what changed:

```bash
git status --short
git diff --stat
git log --oneline -5
```

Stage everything (all changes, including new files and deletions):

```bash
git add -A
```

Write a concise commit message in the repo's conventional-commit style
(`feat(scope):`, `fix(scope):`, `docs(scope):`, `refactor(scope):`, ...),
summarizing what was changed. If the change is large, group it logically
but keep the message concise. Then commit and push to `origin`:

```bash
git commit -m "<type>(<scope>): <summary>"
git push
```

If the current branch has no upstream yet, set it with
`git push -u origin <current-branch>`.

Confirm the push succeeded (e.g. `git status` shows nothing pending, or the
push output reports success) and report the commit hash / branch to the user.
