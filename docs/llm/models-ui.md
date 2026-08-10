# Models UI — Technical Specification

Rust TUI application at `dev/models-ui/` that manages Hugging Face model
downloads for Story Engine. Invoked via `make models-ui`.

## Architecture

Single-binary Rust app (`src/main.rs`), ~550 lines. No config files; it
derives everything from the project itself at startup.

### Data flow

1. **Model discovery**: regex-parses `models.py` at the project root —
   `MODEL_METADATA` (key, name, type, size, repo_id) and `MODEL_PATHS`
   (type → folder). There is no duplicated model list; `models.py` remains
   the single source of truth.
2. **Status detection**: recursively sums bytes under
   `models/<type>/<key>`. Status = `Pending` (0 bytes), `Partial`
   (< 85% of expected size parsed from the `size` metadata string), or
   `Completed`.
3. **Downloads**: shells out to `hf download <repo_id> --local-dir <dest>`
   (binary resolved as `$HF_BIN` → `.venv/bin/hf` → `hf` on PATH).
   One child process at a time; checked models form a FIFO queue.
4. **Progress**: the UI thread polls `dir_size(dest)` every 2s. Speed is
   computed from a 30s sliding window of `(Instant, bytes)` samples;
   ETA = remaining bytes / speed. This is deliberately independent of the
   hf CLI's own progress output, which is suppressed on non-TTY stderr.
5. **Cancellation**: the child PID is stored in `Arc<Mutex<Option<u32>>>`;
   pressing `x` sends SIGTERM (`kill <pid>`). A `cancelling` flag makes the
   resulting non-zero exit map to `Partial` (resumable) instead of `Failed`,
   and suppresses auto-starting the next queued item.

### Threading model

- **Main thread**: ratatui render loop + crossterm event polling (200ms tick).
- **Per download**: one thread spawns/waits on the `hf` child; a second
  thread reads the child's stderr byte-wise (splitting on `\r` and `\n` to
  handle progress-bar rewrites) and forwards lines over an
  `mpsc::channel` as `DlMsg::Line`; completion sends `DlMsg::Done(idx, ok)`.
- If hf emits no output for >5s, the main thread synthesizes a status line
  from the disk-size polling so the output pane always shows activity.

### Crates

| Crate | Purpose |
|-------|---------|
| `ratatui` 0.29 | TUI widgets: List (checkboxes/status), Gauge (progress), Paragraph |
| `crossterm` 0.28 | Terminal raw mode, alternate screen, key events |
| `regex` 1 | Parsing `models.py` and size strings ("~60GB", "1.2GB") |

Everything else is std (`process`, `thread`, `sync::mpsc`, `fs`).

## CLI

```
models-ui [--list] [project_root]
```

- `--list`: non-interactive; prints parsed models with status/sizes and
  exits. Useful for scripts and for verifying the models.py parser.
- `project_root` defaults to the current directory; must contain
  `models.py` and `models/`.

## Environment set on the hf child

- `PYTHONUNBUFFERED=1`, `HF_HUB_DISABLE_PROGRESS_BARS=0`, `FORCE_COLOR=0` —
  to coax progress output through the pipe.

## Invariants / constraints

- At most one active download (`App.downloading: Option<usize>`).
- Only `Pending | Partial | Failed` items are checkable/queueable.
- Manual stop (`x`) pauses the queue; `d` resumes it.
- Statuses are recomputed from disk on `r` and after each download ends.
- Expected sizes come from the `size` strings in `models.py`; progress
  ratio and ETA are estimates, capped at 100%.
