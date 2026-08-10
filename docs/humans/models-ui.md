# Models UI — Interactive Model Download Manager

A small terminal app (TUI) to manage downloading the large models used by
Story Engine — especially the huge image-to-video ones — one at a time,
with visible progress.

## Launch

```bash
make models-ui
```

(Requires Rust/cargo. First run compiles the app; later runs start instantly.)

## What you'll see

A list of every model defined in `models.py`, with its status detected from
the `models/` folder on disk:

- **✔ completed** — folder size looks complete (≥ ~85% of the expected size)
- **partial** — some data on disk, but incomplete (e.g. an interrupted
  download). Has a checkbox; downloading it **resumes** where it left off
- **pending** — nothing on disk yet. Has a checkbox

Below the list: a progress bar for the active download (with size, speed and
ETA) and an output line showing what's happening.

## Keys

| Key | Action |
|-----|--------|
| `↑` / `↓` (or `j`/`k`) | Move selection |
| `SPACE` | Check/uncheck a pending, partial or failed model |
| `d` or `ENTER` | Download all checked models, one at a time |
| `x` | Stop the active download (keeps partial data; resume later) |
| `r` | Re-scan the `models/` folder |
| `q` | Quit (`Shift+Q` forces quit during a download) |

## Notes

- Downloads run **sequentially**: check several boxes and they queue up.
- Stopping with `x` pauses the queue too — press `d` to continue it.
- Downloads are resumable: partial models keep their data and continue from
  where they stopped.
- Progress/speed is measured from the folder size on disk, so it works even
  when the underlying `hf` CLI prints nothing (it hides its progress bars
  when not attached to a terminal). If the `hf output` pane shows a
  synthesized line like `Wan 2.2 I2V A14B: 12.3GB on disk @ 45MB/s`,
  that's normal — it means the download is progressing.
- ETA accuracy depends on the `size` values declared in `models.py`.

## Troubleshooting

- **"failed to start hf"** — the app looks for `.venv/bin/hf` in the project
  root, then `hf` on PATH. Override with the `HF_BIN` env var.
- **Gated models (e.g. FLUX.1-dev)** — accept the license on Hugging Face and
  make sure you're logged in (`hf auth login`) before downloading.
- **A model shows "partial" but is actually fine** — the expected sizes in
  `models.py` are rough estimates; fix the `size` field there.
