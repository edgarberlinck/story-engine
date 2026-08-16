---
name: publish-to-git
description: Publish the current work to git. Runs tests, lint, coverage (>= 80%), and build, then commits ALL changes and pushes. Use when the user says "send it to git", "you can push the code", "publish it", "commit and push", "push everything", or any similar request to release current work to the repository.
---

# Publish to Git

Run every step below, in order. If any step fails, STOP and report the
failure to the user — do not commit or push until the failing check is
resolved. Do not skip a step unless the user explicitly asks.

## 1. Tests must pass

Run the full test suite:

```bash
make test
```

This runs `python -m unittest discover -s . -p "test_*.py" -v`.
If it reports anything other than `OK`, fix the failures (or report them)
and re-run until green.

## 2. Lint must pass

Run the linter:

```bash
make lint
```

This runs `flake8 .`. Fix any reported violations (or report them) before
continuing. Prefer `make format` (black) to auto-fix style issues, then
re-run lint.

## 3. Coverage must be at least 80%

Ensure `coverage` is available in the virtualenv (it is not part of
`requirements.txt`):

```bash
.venv/bin/pip install coverage
```

Then measure coverage while running the test suite:

```bash
.venv/bin/python -m coverage erase
.venv/bin/python -m coverage run --source . --omit=".venv/*,*/test_*.py,test_*.py" -m unittest discover -s . -p "test_*.py"
.venv/bin/python -m coverage report --fail-under=80
```

If the report shows coverage below 80%, STOP and report it — do not
commit. The `--fail-under=80` flag makes the command exit non-zero on
insufficient coverage.

Clean up the coverage artifacts afterwards:

```bash
.venv/bin/python -m coverage erase
```

## 4. Build must succeed

This is a pure-Python project with no packaging config, so the build check
is byte-compiling every module:

```bash
.venv/bin/python -m compileall -q .
```

Any syntax/compile error fails this step. (If a real build system is added
later — `pyproject.toml`, wheels, etc. — replace this step with the actual
build command.)

## 5. Commit ALL changes and push

Inspect what changed:

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
but keep the message concise. Then commit and push:

```bash
git commit -m "<type>(<scope>): <summary>"
git push origin <current-branch>
```

Confirm the push succeeded (e.g. `git status` shows nothing pending, or the
push output reports success) and report the commit hash / branch to the
user.